"""
# @Time    : 2022/8/3 10:02
# @File    : pytorch2onnx.py
# @Author  : rezheaiba
"""
import torch.nn

from model_v3_multclassifier import MobileNetV3

net = MobileNetV3(dilated=True, num_classes_1=2, num_classes_2=5, num_classes_3=2, num_classes_4=2,
                  arch='mobilenet_v3_small',
                  reduced_tail=True, width_mult=0.5).cpu()
model_weight_path = "../model_v3_mult.pth"
net.load_state_dict(torch.load(model_weight_path))  # 模型加载到自己的device上
net.cuda()
net.eval()

input_names = ['input']
output_names = ['output1', 'output2', 'output3', 'output4']

# model_script = torch.jit.script(net)

x = torch.randn(1, 3, 256, 256, requires_grad=True).cuda()
# y = net(x)

torch.onnx.export(net, x, "../model_v3_mult.onnx", opset_version=10,
                  input_names=input_names,
                  output_names=output_names, verbose='True')
