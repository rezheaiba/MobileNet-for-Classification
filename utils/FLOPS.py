"""
# @Time    : 2022/12/20 15:35
# @File    : FLOPS.py
# @Author  : rezheaiba
"""
import torch
from torchvision.models import *
from model_v3 import MobileNetV3
# from model_v3_multclassifier import MobileNetV3
from thop import profile
from thop import clever_format

model = mobilenet_v2(pretrained=False, width_mult=0.3, num_classes=7)
model = MobileNetV3(width_mult=0.5, num_classes=7, arch='mobilenet_v3_small',reduced_tail=True,dilated=True)

input = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(input, ))
print('parmas:{}, flops:{}'.format(params, flops))
print(
        "%s | %.2f M | %.2f G" % ('mobilenet_v2', params / (1000 ** 2), flops / (1000 ** 3))
    )
flops, params = clever_format([flops, params], '%3.f')
print('parmas:{}, flops:{}'.format(params, flops))
# torch.save(model.state_dict(), './v2.pth')
torch.save(model.state_dict(), './v3.pth')


'''mobilenet_v2 width_mult=0.25 | 1.52 M | 0.06 G
parmas:  2M, flops: 60M'''

'''mobilenet_v3 width_mult=0.25 | 0.37 M | 0.01 G
parmas:374K, flops: 14M'''




'''mobilenet_v2 | 3.50 M | 0.33 G
parmas:  4M, flops:327M'''

'''mobilenet_v3_samll | 2.54 M | 0.06 G
parmas:  3M, flops: 62M'''

'''mobilenet_v3_large | 5.48 M | 0.23 G
parmas:  5M, flops:235M'''