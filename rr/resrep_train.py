from torch.utils.tensorboard import SummaryWriter
from base_config import BaseConfigByEpoch
from model_map import get_model_fn
from data.data_factory import create_dataset, load_cuda_data, num_iters_per_epoch
from torch.nn.modules.loss import CrossEntropyLoss
from utils.pyt_utils import ensure_dir
from utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
from tqdm import tqdm
import torchvision
import time
from utils.lr_scheduler import get_lr_scheduler
import os
from ndp_test import val_during_train
from utils.checkpoint import get_last_checkpoint
from rr.resrep_util import *
from rr.resrep_convert import *
from utils.engine import Engine
from rr.resrep_scripts import *
from kd_losses import *

TRAIN_SPEED_START = 0.1
TRAIN_SPEED_END = 0.2

COLLECT_TRAIN_LOSS_EPOCHS = 3

TEST_BATCH_SIZE = 100

CONVERSION_EPSILON = 1e-5
DEPS = rc_origin_deps_flattened(9)
# def train_one_step(compactor_mask_dict, resrep_config:ResRepConfig,
#                    net, data, label, optimizer, criterion, if_accum_grad = False,
#                    gradient_mask_tensor = None):
def train_one_step(resrep_config:ResRepConfig,
                   net, data, label, optimizer, criterion, if_accum_grad = False,
                   gradient_mask_tensor = None,criterion2=None,mid_feature=None,idx=None):
    # pred = net(data)#原网络
    # mid_feature2=pred
    mid_feature=mid_feature
    criterion2=criterion2
    mid_feature2,pred=get_fea_out_dic2(model=net,data=data,idx=idx)
    criten =COSFEATURE()
    # print(mid_feature.size())
    loss= criten(mid_feature[0],mid_feature2[0])
    # m_slist = []
    # for name,value in m_s_dict.items():
    #     m_slist.append(abs(value))
    # loss = criterion(pred, label)
    loss.backward()
    #TODO:改一个LOSS，截止某一层的feature误差
    #E #for name,value in m_s_dict.items():
    #     lasso_grad = value.data/(value.data**2)**0.5
    #     # if
    #     value.grad.data.add_(1e-2,lasso_grad)
    # for compactor_param, mask in compactor_mask_dict.items():#魔改的梯度mask
    #     compactor_param.grad.data = mask * compactor_param.grad.data
    #     lasso_grad = compactor_param.data * ((compactor_param.data ** 2).sum(dim=(1, 2, 3), keepdim=True) ** (-0.5))
    #                                             #欧几里德范数
    #     compactor_param.grad.data.add_(resrep_config.lasso_strength, lasso_grad)#1e-4


    if not if_accum_grad:
        if gradient_mask_tensor is not None:
            for name, param in net.named_parameters():
                if name in gradient_mask_tensor:
                    param.grad = param.grad * gradient_mask_tensor[name]
        optimizer.step()
        optimizer.zero_grad()
    acc, acc5 = torch_accuracy(pred, label, (1,5))
    # acc=0
    # acc5=0
    # loss=0
    return acc, acc5, loss

def sgd_optimizer(cfg, resrep_config:ResRepConfig, model, no_l2_keywords, use_nesterov, keyword_to_lr_mult):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        if "bias" in key or "bn" in key or "BN" in key:
            weight_decay = cfg.weight_decay_bias
            print('set weight_decay_bias={} for {}'.format(weight_decay, key))
        for kw in no_l2_keywords:
            if kw in key:
                weight_decay = 0
                print('NOTICE! weight decay = 0 for ', key, 'because {} in {}'.format(kw, key))
                break
        if 'bias' in key:
            apply_lr = 2 * lr
        else:
            apply_lr = lr
        if keyword_to_lr_mult is not None:
            for keyword, mult in keyword_to_lr_mult.items():
                if keyword in key:
                    apply_lr *= mult
                    print('multiply lr of {} by {}'.format(key, mult))
                    break
        if 'compactor' in key:
            use_momentum = resrep_config.compactor_momentum
            print('momentum {} for {}'.format(use_momentum, key))
        else:
            use_momentum = cfg.momentum
        params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay, "momentum": use_momentum}]
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.momentum, nesterov=use_nesterov)
    return optimizer



def get_optimizer(cfg, resrep_config, model, no_l2_keywords, use_nesterov=False, keyword_to_lr_mult=None):
    return sgd_optimizer(cfg, resrep_config, model, no_l2_keywords, use_nesterov=use_nesterov, keyword_to_lr_mult=keyword_to_lr_mult)

def get_criterion(cfg):
    return CrossEntropyLoss()
def calculate_run_flops(model:nn.Module):
    result = []
    idxx = -1
    hdic=[32]*19+[16]*19+[8]*19
    for child_model in model.modules():
        if type(child_model)==Mlayer:
            if child_model.mask>0.9:
                idxx += 1
                result.append(calculate_res56_block_flops(DEPS[2*idxx],DEPS[2*idxx+1],DEPS[2*idxx+2],hdic[2*idxx],hdic[2*idxx]))
            else :
                result.append(
                    calculate_mobv3_block_flops(DEPS[2 * idxx], DEPS[2 * idxx + 1], DEPS[2 * idxx + 2], hdic[2*idxx], hdic[2*idxx]))
    return np.sum(result)

def get_fea_out_dic1(model,data,idx):
    features_out_hook = []
    idx=idx
    def hook(module, fea_in, fea_out):
        features_out_hook.append(fea_out.clone())
        return None
    handle=model.__getattr__('stage{}'.format(idx//9+1)).\
        __getattr__('block{}'.format(idx%9)).__getattr__('conv2').register_forward_hook(hook)
    out = model(data)
    handle.remove()
    return features_out_hook
def get_fea_out_dic2(model,data,idx):
    features_out_hook = []
    idx=idx
    def hook(module, fea_in, fea_out):
        features_out_hook.append(fea_out.clone())
        return None
    handle=model.__getattr__('stage{}'.format(1+idx//9)).\
        __getattr__('block{}'.format(idx%9)).__getattr__('mob1branch').register_forward_hook(hook)
    out = model(data)
    handle.remove()
    return features_out_hook,out
def resrep_train_main(
        local_rank,
        cfg:BaseConfigByEpoch, resrep_config:ResRepConfig, resrep_builder,
        net=None, train_dataloader=None, val_dataloader=None, show_variables=False,
               init_hdf5=None, no_l2_keywords='depth', gradient_mask=None, use_nesterov=False,
               load_weights_keyword=None,
               keyword_to_lr_mult=None,
               auto_continue=False,save_hdf5_epochs=5,mask_lis=None):

    if no_l2_keywords is None:
        no_l2_keywords = []
    if type(no_l2_keywords) is not list:
        no_l2_keywords = [no_l2_keywords]

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.tb_dir)
    with Engine(local_rank=local_rank) as engine:
        engine.setup_log(
            name='train', log_dir=cfg.output_dir, file_name='log.txt')

        # ----------------------------- build model ------------------------------
        if net is None:

            net_fn = get_model_fn(cfg.dataset_name, cfg.network_type)
            model = net_fn(cfg, resrep_builder,mask_lis=[0]+[1]*26)#上一行最后return SRCNet需要cfg和builder参数
                                                #这里的builder和base的builder不一样
            model_no_mask = net_fn(cfg, resrep_builder,mask_lis=[1] * 27)
        else:
            model = net
            net_fn = get_model_fn(cfg.dataset_name, cfg.network_type)
            model_no_mask = net_fn(cfg, resrep_builder, mask_lis=[1] * 27)
        model = model.cuda()
        model_no_mask=model_no_mask.cuda()
        # ----------------------------- model done ------------------------------

        # ---------------------------- prepare data -------------------------
        if train_dataloader is None:
            train_data = create_dataset(cfg.dataset_name, cfg.dataset_subset,
                                            cfg.global_batch_size, distributed=engine.distributed)
        if cfg.val_epoch_period > 0 and val_dataloader is None:
            val_data = create_dataset(cfg.dataset_name, 'val',
                                      global_batch_size=100, distributed=False)
        engine.echo('NOTE: Data prepared')
        engine.echo('NOTE: We have global_batch_size={} on {} GPUs, the allocated GPU memory is {}'.format(cfg.global_batch_size, torch.cuda.device_count(), torch.cuda.memory_allocated()))
        # ----------------------------- data done --------------------------------


        # ------------------------ parepare optimizer, scheduler, criterion -------
        optimizer = get_optimizer(cfg, resrep_config, model,
                                  no_l2_keywords=no_l2_keywords, use_nesterov=use_nesterov, keyword_to_lr_mult=keyword_to_lr_mult)
        scheduler = get_lr_scheduler(cfg, optimizer)
        criterion = get_criterion(cfg).cuda()
        criterion2 = torch.nn.MSELoss()
        # --------------------------------- done -------------------------------
        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer)

        #   finish.hdf5 exists, do conversion and return
        if os.path.exists(os.path.join(cfg.output_dir, 'finish.hdf5')):
            engine.load_hdf5(os.path.join(cfg.output_dir, 'finish.hdf5'), load_weights_keyword=load_weights_keyword)
            if 'mi1' in cfg.network_type:
                compactor_convert_mi1(model=model, origin_deps=cfg.deps, thresh=CONVERSION_EPSILON,
                                      save_path=os.path.join(cfg.output_dir,
                                                             'finish_converted.hdf5'))
            else:
                compactor_convert(model=model, origin_deps=cfg.deps, thresh=CONVERSION_EPSILON,
                                  pacesetter_dict=resrep_config.pacesetter_dict,
                                  succ_strategy=resrep_config.succeeding_strategy,
                                  save_path=os.path.join(cfg.output_dir, 'finish_converted.hdf5'))

            return

        if engine.distributed:
            torch.cuda.set_device(local_rank)
            engine.echo('Distributed training, device {}'.format(local_rank))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank],
                broadcast_buffers=False, )
        else:
            assert torch.cuda.device_count() == 1
            engine.echo('Single GPU training')

        if cfg.init_weights:
            engine.load_checkpoint(cfg.init_weights)
        if init_hdf5:
            engine.load_hdf5(init_hdf5, load_weights_keyword=load_weights_keyword)

        if auto_continue:
            assert cfg.init_weights is None
            engine.load_checkpoint(get_last_checkpoint(cfg.output_dir))
        if show_variables:
            engine.show_variables()

        # ------------ do training ---------------------------- #
        engine.log("\n\nStart training with pytorch version {}".format(torch.__version__))

        iteration = engine.state.iteration
        iters_per_epoch = num_iters_per_epoch(cfg)
        max_iters = iters_per_epoch * cfg.max_epochs
        tb_writer = SummaryWriter(cfg.tb_dir)
        tb_tags = ['Top1-Acc', 'Top5-Acc', 'Loss']

        model.train()
        for name,param in model.named_parameters():
            if not 'mob1branch' in name and not 'm_s' in name:
                param.requires_grad=False

        for name,param in model_no_mask.named_parameters():
            if not 'mob1branch' in name and not 'm_s' in name:
                param.requires_grad=False
        done_epochs = iteration // iters_per_epoch
        last_epoch_done_iters = iteration % iters_per_epoch

        if done_epochs == 0 and last_epoch_done_iters == 0:
            engine.save_hdf5(os.path.join(cfg.output_dir, 'init.hdf5'))

        recorded_train_time = 0
        recorded_train_examples = 0

        collected_train_loss_sum = 0
        collected_train_loss_count = 0

        if gradient_mask is not None:
            gradient_mask_tensor = {}
            for name, value in gradient_mask.items():
                gradient_mask_tensor[name] = torch.Tensor(value).cuda()
        else:
            gradient_mask_tensor = None

        compactor_mask_dict = get_compactor_mask_dict(model)

        idx = -1
        for epoch in range(done_epochs, cfg.max_epochs):
            if epoch%(cfg.max_epochs//30)==0 and idx < 26:
                idx += 1
                # 保存模型参数
                torch.save(model.state_dict(), 'model_params.pth')
                model = net_fn(cfg, resrep_builder, mask_lis=[0] * (idx+1) + [1] * (26- idx))

                # 加载模型参数


                model.load_state_dict(torch.load('model_params.pth'))
                model = model.cuda()
                for name, param in model.named_parameters():
                    if not 'mob1branch' in name and not 'm_s' in name:
                        param.requires_grad = False
                    if idx==0:
                        continue
                    elif 'stage{}.block{}'.format(idx//9+1,(idx-1)%9) in name:
                        param.requires_grad =False


            # if epoch == 0 and engine.local_rank == 0:
            #     val_during_train(epoch=epoch, iteration=iteration, tb_tags=tb_tags, engine=engine, model=model,
            #                  val_data=val_data, criterion=criterion,
            #                      descrip_str='Begin',
            #                  dataset_name=cfg.dataset_name, test_batch_size=TEST_BATCH_SIZE, tb_writer=tb_writer)

            if engine.distributed and hasattr(train_data, 'train_sampler'):
                train_data.train_sampler.set_epoch(epoch)

            if epoch == done_epochs:
                pbar = tqdm(range(iters_per_epoch - last_epoch_done_iters))
            else:
                pbar = tqdm(range(iters_per_epoch))

            top1 = AvgMeter()
            top5 = AvgMeter()
            losses = AvgMeter()
            discrip_str = 'Epoch-{}/{}'.format(epoch, cfg.max_epochs)
            pbar.set_description('Train' + discrip_str)


            for _ in pbar:

                if iteration > resrep_config.before_mask_iters:
                    total_iters_in_compactor_phase = iteration - resrep_config.before_mask_iters
                    if total_iters_in_compactor_phase > 0 and (total_iters_in_compactor_phase % resrep_config.mask_interval == 0):
                        print('update mask at iter ', iteration)#经过一定阶段才会mask


                        # for name, param in model.named_parameters():
                        #     if not 'mob3branch' in name and not 'm_s' in name:
                        #         param.requires_grad = False
                            # if 'block0' in name:
                            #     param.requires_grad=False
                        # resrep_mask_model(origin_deps=cfg.deps, resrep_config=resrep_config, model=model)#修改了compactor的mask参数
                        # compactor_mask_dict = get_compactor_mask_dict(model=model)
                        # layer_mask_model()

                        '''
                        1.计算flops
                        
                        '''
                        print('='*20+'run_flops{}'.format(calculate_run_flops(model))+'='*20)
                        unmasked_deps = resrep_get_unmasked_deps(origin_deps=cfg.deps, model=model, pacesetter_dict=resrep_config.pacesetter_dict)
                        engine.log('iter {}, unmasked deps {}'.format(iteration, list(unmasked_deps)))
                        if total_iters_in_compactor_phase == resrep_config.mask_interval:
                            engine.save_hdf5(os.path.join(cfg.output_dir, 'before_first_mask.hdf5'))


                start_time = time.time()
                data, label = load_cuda_data(train_data, dataset_name=cfg.dataset_name)


                    # load_cuda_data(train_dataloader, cfg.dataset_name)
                data_time = time.time() - start_time

                if_accum_grad = ((iteration % cfg.grad_accum_iters) != 0)

                train_net_time_start = time.time()
                # m_s_dict = get_m_s_dict(model=model)

                #TODO:model_no_mask进行前向并获取相应的feature
                fea_out_dic = get_fea_out_dic1(model=model_no_mask,data=data,idx=idx)
                # fea_out_dic= model_no_mask(data)
                acc, acc5, loss = train_one_step(resrep_config, model, data, label, optimizer,
                                                                                  criterion,
                                                                                  if_accum_grad, gradient_mask_tensor=gradient_mask_tensor,criterion2=criterion2,mid_feature=fea_out_dic,idx=idx)

                train_net_time_end = time.time()

                if iteration > TRAIN_SPEED_START * max_iters and iteration < TRAIN_SPEED_END * max_iters:
                    recorded_train_examples += cfg.global_batch_size
                    recorded_train_time += train_net_time_end - train_net_time_start

                scheduler.step()

                for module in model.modules():
                    if hasattr(module, 'set_cur_iter'):
                        module.set_cur_iter(iteration)

                # if iteration % cfg.tb_iter_period == 0 and engine.world_rank == 0:

                if iteration % cfg.tb_iter_period == 0:
                    # for name, param in model.named_parameters():
                    #     if 'compactor' in name:
                    #         tb_writer.add_image(name + "_model1", param.clone().reshape(1,param.size()[0],-1).cpu().data.numpy(), iteration)
                    for tag, value in zip(tb_tags, [acc.item(), acc5.item(), loss.item()]):
                        tb_writer.add_scalars(tag, {'Train': value}, iteration)
                    for name,param in model.named_parameters():
                        if 'mask' in name:
                            tb_writer.add_scalars(name,{'mask':param},iteration)

                top1.update(acc.item())
                top5.update(acc5.item())
                losses.update(loss.item())

                if epoch >= cfg.max_epochs - COLLECT_TRAIN_LOSS_EPOCHS:
                    collected_train_loss_sum += loss.item()
                    collected_train_loss_count += 1

                pbar_dic = OrderedDict()
                pbar_dic['data-time'] = '{:.2f}'.format(data_time)
                pbar_dic['cur_iter'] = iteration
                pbar_dic['lr'] = scheduler.get_lr()[0]
                pbar_dic['top1'] = '{:.5f}'.format(top1.mean)
                pbar_dic['top5'] = '{:.5f}'.format(top5.mean)
                pbar_dic['loss'] = '{:.5f}'.format(losses.mean)
                pbar.set_postfix(pbar_dic)

                iteration += 1

                if iteration >= max_iters or iteration % cfg.ckpt_iter_period == 0:
                    engine.update_iteration(iteration)
                    if (not engine.distributed) or (engine.distributed and engine.world_rank == 0):
                        engine.save_and_link_checkpoint(cfg.output_dir)
                if iteration % 1000 == 0:
                    origin_flops = resrep_config.flops_func(cfg.deps)
                    remain_deps = get_deps_if_prune_low_metric(origin_deps=cfg.deps, model=model,
                                                               threshold=CONVERSION_EPSILON,
                                                               pacesetter_dict=resrep_config.pacesetter_dict)
                    remain_flops = resrep_config.flops_func(remain_deps)
                    engine.log('iter {}, thres {}, remain deps {}, FLOPs {:.4f}'.format(iteration, CONVERSION_EPSILON,
                                                                                        remain_deps,
                                                                                        remain_flops / origin_flops))
                if iteration >= max_iters:
                    break

            engine.update_iteration(iteration)
            engine.save_latest_ckpt(cfg.output_dir)

            if (epoch + 1) % save_hdf5_epochs == 0:
                engine.save_hdf5(os.path.join(cfg.output_dir, 'epoch-{}.hdf5'.format(epoch)))

            if local_rank == 0 and \
                    cfg.val_epoch_period > 0 and (epoch >= cfg.max_epochs - 10 or epoch % cfg.val_epoch_period == 0):
                val_during_train(epoch=epoch, iteration=iteration, tb_tags=tb_tags, engine=engine, model=model,
                                 val_data=val_data, criterion=criterion, descrip_str=discrip_str,
                                 dataset_name=cfg.dataset_name, test_batch_size=TEST_BATCH_SIZE, tb_writer=tb_writer)

            if iteration >= max_iters:
                break

        #   do something after the training
        # if recorded_train_time > 0:
        #     exp_per_sec = recorded_train_examples / recorded_train_time
        # else:
        #     exp_per_sec = 0
        # engine.log(
        #     'TRAIN speed: from {} to {} iterations, batch_size={}, examples={}, total_net_time={:.4f}, examples/sec={}'
        #     .format(int(TRAIN_SPEED_START * max_iters), int(TRAIN_SPEED_END * max_iters), cfg.global_batch_size,
        #             recorded_train_examples, recorded_train_time, exp_per_sec))
        # if cfg.save_weights:
        #     engine.save_checkpoint(cfg.save_weights)
        #     print('NOTE: training finished, saved to {}'.format(cfg.save_weights))
        # engine.save_hdf5(os.path.join(cfg.output_dir, 'finish.hdf5'))
        # if collected_train_loss_count > 0:
        #     engine.log('TRAIN LOSS collected over last {} epochs: {:.6f}'.format(COLLECT_TRAIN_LOSS_EPOCHS,
        #                                                                          collected_train_loss_sum / collected_train_loss_count))
        # if 'mi1' in cfg.network_type:
        #     compactor_convert_mi1(model=model, origin_deps=cfg.deps, thresh=CONVERSION_EPSILON,
        #                           save_path=os.path.join(cfg.output_dir, 'finish_converted.hdf5'))
        # else:
        #     compactor_convert(model=model, origin_deps=cfg.deps, thresh=CONVERSION_EPSILON,
        #                       pacesetter_dict=resrep_config.pacesetter_dict,
        #                       succ_strategy=resrep_config.succeeding_strategy,
        #                       save_path=os.path.join(cfg.output_dir, 'finish_converted.hdf5'))


