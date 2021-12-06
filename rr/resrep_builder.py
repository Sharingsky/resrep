from rr.resrep_config import ResRepConfig
from builder import ConvBuilder
from base_config import BaseConfigByEpoch
import torch.nn as nn
import torch
from rr.compactor import CompactorLayer
class Mlayer(nn.Module):
    def __init__(self,mlayeridx=0,threshhold=1e-4):
        super(Mlayer, self).__init__()
        # m_s = torch.ones([1,in_channel,1,1],requires_grad=True)
        m_s = torch.ones(1)
        self.m_s = torch.nn.Parameter(m_s,requires_grad=True)
        self.mask =torch.ones(1,dtype=torch.int32).cuda()
        self.register_buffer('M_mask',self.mask)
        self.mlayeridx=mlayeridx
        self.is_masked = False
        self.threshhold=threshhold
    def forward(self, input):
        x = input * (self.m_s**2/(self.m_s**2+self.threshhold))
        return x

class LayerPruneBuilder(ConvBuilder):
    def __init__(self, base_config:BaseConfigByEpoch, resrep_config:ResRepConfig, mode='train'):
        super(LayerPruneBuilder, self).__init__(base_config=base_config)
        self.resrep_config = resrep_config
        assert mode in ['train', 'deploy']
        self.mode = mode

    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', use_original_conv=False):
        self.cur_conv_idx += 1
        assert type(kernel_size) is int
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.cur_mask_idx +=1

        if self.mode == 'deploy':
            se = self.Sequential()
            se.add_module('conv', super(LayerPruneBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                                    kernel_size=kernel_size, stride=stride,
                                                                    padding=padding, dilation=dilation, groups=groups,
                                                                    padding_mode=padding_mode,
                                                                    bias=True))
            return se

        if use_original_conv or self.cur_conv_idx not in self.resrep_config.target_layers:
            self.cur_conv_idx -= 1
            print('layer {}, use original conv'.format(self.cur_conv_idx + 1))
            return super(LayerPruneBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels,
                                                       kernel_size=kernel_size, stride=stride,
                                                       padding=padding, dilation=dilation, groups=groups,
                                                       padding_mode=padding_mode)

        else:

            se = self.Sequential()
            se_main = self.Sequential()
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                                   padding_mode=padding_mode)

            se_main.add_module('conv', conv_layer)
            bn_layer = self.BatchNorm2d(num_features=out_channels)
            se_main.add_module('bn', bn_layer)
            se_main.add_module('m_s_layer', Mlayer(self.cur_mask_idx))
            se.add_module('se_main', se_main)
            se.add_module('shot_conv1',nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,
                                                 stride=stride))

            return se
class Conv1ShotBuilder(ConvBuilder):

    def __init__(self, base_config:BaseConfigByEpoch, resrep_config:ResRepConfig, mode='train'):
        super(Conv1ShotBuilder, self).__init__(base_config=base_config)
        self.resrep_config = resrep_config
        assert mode in ['train', 'deploy']
        self.mode = mode

    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               padding_mode='zeros', use_original_conv=False):
        self.cur_conv_idx += 1
        assert type(kernel_size) is int
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        if self.mode == 'deploy':
            se = self.Sequential()
            se.add_module('conv', super(Conv1ShotBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                                    padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode,
                                                                    bias=True))
            return se

        if use_original_conv or self.cur_conv_idx not in self.resrep_config.target_layers:
            self.cur_conv_idx -= 1
            print('layer {}, use original conv'.format(self.cur_conv_idx + 1))
            return super(Conv1ShotBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)

        else:
            se = self.Sequential()
            se_main = self.Sequential()
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                                   padding_mode=padding_mode)
            se_main.add_module('conv', conv_layer)
            bn_layer = self.BatchNorm2d(num_features=out_channels)
            se_main.add_module('bn', bn_layer)
            se_main.add_module('compactor', CompactorLayer(num_features=out_channels, conv_idx=self.cur_conv_idx))
            se.add_module('se_main',se_main)
            se.add_module('conv1shot',nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride))
            return se
class LayerMaskBuilder(ConvBuilder):

    def __init__(self, base_config:BaseConfigByEpoch, resrep_config:ResRepConfig, mode='train'):
        super(LayerMaskBuilder, self).__init__(base_config=base_config)
        self.resrep_config = resrep_config
        assert mode in ['train', 'deploy']
        self.mode = mode

    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               padding_mode='zeros', use_original_conv=False):
        self.cur_conv_idx += 1
        assert type(kernel_size) is int
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        if self.mode == 'deploy':
            se = self.Sequential()
            se.add_module('conv', super(LayerMaskBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                                    padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode,
                                                                    bias=True))
            return se

        if use_original_conv or self.cur_conv_idx not in self.resrep_config.target_layers:
            self.cur_conv_idx -= 1
            print('layer {}, use original conv'.format(self.cur_conv_idx + 1))
            return super(LayerMaskBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)

        else:
            se = self.Sequential()
            se_main = self.Sequential()
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                                   padding_mode=padding_mode)
            se_main.add_module('conv', conv_layer)
            bn_layer = self.BatchNorm2d(num_features=out_channels)
            se_main.add_module('bn', bn_layer)
            se_main.add_module('compactor', CompactorLayer(num_features=out_channels, conv_idx=self.cur_conv_idx))
            se.add_module('se_main',se_main)
            se.add_module('mlayer',Mlayer(in_channel=in_channels,out_channel=out_channels))
            return se
class ResRepBuilder(ConvBuilder):

    def __init__(self, base_config:BaseConfigByEpoch, resrep_config:ResRepConfig, mode='train'):
        super(ResRepBuilder, self).__init__(base_config=base_config)
        self.resrep_config = resrep_config
        assert mode in ['train', 'deploy']
        self.mode = mode

    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               padding_mode='zeros', use_original_conv=False):
        self.cur_conv_idx += 1
        assert type(kernel_size) is int
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        if self.mode == 'deploy':
            se = self.Sequential()
            se.add_module('conv', super(ResRepBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                                    padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode,
                                                                    bias=True))
            return se

        if use_original_conv or self.cur_conv_idx not in self.resrep_config.target_layers:
            self.cur_conv_idx -= 1
            print('layer {}, use original conv'.format(self.cur_conv_idx + 1))
            return super(ResRepBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)

        else:

            se = self.Sequential()
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                                   padding_mode=padding_mode)
            se.add_module('conv', conv_layer)
            bn_layer = self.BatchNorm2d(num_features=out_channels)
            se.add_module('bn', bn_layer)
            se.add_module('compactor', CompactorLayer(num_features=out_channels, conv_idx=self.cur_conv_idx))
            print('use compactor on conv {} with kernel size {}'.format(self.cur_conv_idx, kernel_size))
            return se