"""
# @Time    : 2022/8/3 10:02
# @File    : pytorch2onnx.py
# @Author  : rezheaiba
"""
import torch
import torch.nn
import onnx
from model_v2_25_softmax import MobileNetV2

net = MobileNetV2(4)  # place365提供的权重已经把全连接改成了365个特征
model_weight_path = "../weight/MobileNetV2_level2_best.pth"
net.load_state_dict(torch.load(model_weight_path))  # 模型加载到自己的device上
net.cuda()
net.eval()

input_names = ['input']
output_names = ['output']

# model_script = torch.jit.script(net)

x = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()
# y = net(x)

torch.onnx.export(net, x, "../weight/MobileNetV2_level2_best.onnx", opset_version=10,
                  input_names=input_names,
                  output_names=output_names, verbose='True')

