import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torchvision.ops as ops
import time
import torch
from net.strip_pooling import StripPooling
from net.ECAAttention import ECAAttention
from net.CoTAttention import CoTAttention
class CSRNet_deform_var(nn.Module):
    def __init__(self, extra_loss=1, n_deform_layer=1, deform_dilation=2):
        super(CSRNet_deform_var, self).__init__()
        self.dd = deform_dilation
        self.extra_loss = extra_loss
        self.n_dilated = 6 - n_deform_layer
        self.backend_feat_1 = (512, 2)
        self.backend_feat_2 = (512, 2)
        self.backend_feat_3 = (512, 2)
        self.backend_feat_4 = (512, 2)
        self.backend_feat_5 = (128, 2)
        self.backend_feat_6 = (64, 2)
        mod = models.vgg16_bn(pretrained=False)
        mod.load_state_dict(torch.load('/root/autodl-tmp/ShanghaiTech_else/vgg16_bn-6c64b313.pth'))
        self.front_end = nn.Sequential(*(list(list(mod.children())[0].children())[0:33]))
        
        # normal dilated convs
        for j in range(1, self.n_dilated + 1):
            if j == 1:
                in_c = 512
                out_c = self.backend_feat_1[0]
            else:
                in_c = getattr(self, 'backend_feat_{:d}'.format(j-1))[0]
                out_c = getattr(self, 'backend_feat_{:d}'.format(j))[0]

            which_backend_feat = getattr(self, 'backend_feat_{:d}'.format(j))
            setattr(self, 'dconv_{:d}'.format(j), make_layers(which_backend_feat, in_channels=in_c, batch_norm=True))
        
        
        for i in range(self.n_dilated + 1, 7):
            if i == 1:
                in_c = 512
                out_c = self.backend_feat_1[0]
            else:
                in_c = getattr(self, 'backend_feat_{:d}'.format(i-1))[0]
                out_c = getattr(self, 'backend_feat_{:d}'.format(i))[0]

            setattr(self, 'offset_w_{:d}'.format(i), nn.Conv2d(in_channels=in_c, out_channels=2*3*3, kernel_size=3, padding=1))
            setattr(self, 'scale_w_{:d}'.format(i), nn.Conv2d(in_channels=in_c, out_channels=2*3*3, kernel_size=3, padding=1))
            setattr(self, 'deform_{:d}'.format(i), ops.DeformConv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=self.dd, dilation=self.dd))
            setattr(self, 'bn_{:d}'.format(i), nn.BatchNorm2d(out_c))
            
            # Add StripPooling after backend_feat_6
            if i == 6:
                self.ECAAttention = ECAAttention(kernel_size=3)
                setattr(self, 'strip_pooling_{:d}'.format(i), StripPooling(out_c, (20,12),nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True}))
        
        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        for m in self.output_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, out_feat=False):
        img_shape = x.shape
        x = self.front_end(x)
        
        # forward dilated convs
        for i in range(1, self.n_dilated + 1):
            cur_block = getattr(self, 'dconv_{:d}'.format(i))
            x = cur_block(x)
            
        
        x_offset_list = []
        x_offset_scale_list = []
        
        # add loss contrain on the offset
        for j in range(self.n_dilated + 1, 7):
            cur_offset = getattr(self, 'offset_w_{:d}'.format(j))
            cur_offset_scale = getattr(self, 'scale_w_{:d}'.format(j))
            cur_deform = getattr(self, 'deform_{:d}'.format(j))
            cur_bn = getattr(self, 'bn_{:d}'.format(j))
            x_offset = cur_offset(x)
            x_offset_scale = cur_offset_scale(x)
            
            # add offset scale
            x_offset_scale = F.relu_(x_offset_scale)
            x_offset = torch.tanh(x_offset)
            scaled_offset = x_offset_scale * x_offset

            if self.extra_loss:
                x_offset_list.append(scaled_offset)
                x_offset_scale_list.append(x_offset_scale)
            x = F.relu_(cur_bn(cur_deform(x, x_offset)))
            
            # # Apply StripPooling after backend_feat_6
            if i == 6:
                x = self.ECAAttention(x)
                strip_pooling_layer = getattr(self, 'strip_pooling_{:d}'.format(j))
                x = strip_pooling_layer(x)
        
        
        output = self.output_layer(x)

        # add relu for SHB for now, later, SHA also relu
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)

        if self.extra_loss and out_feat:
            return output, x_offset_list
        else:
            return output

def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    cfg = [cfg]
    for v, atrous in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=atrous, dilation=atrous)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



# model = CSRNet_deform_var(n_deform_layer=2)
# print(model)
# input = torch.rand(1, 3, 768, 1024).cuda()
# model = model.cuda()
# output = model(input)
# print(output.shape)
