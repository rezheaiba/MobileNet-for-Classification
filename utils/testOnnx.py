"""
# @Time    : 2022/8/8 18:30
# @File    : testOnnx.py
# @Author  : rezheaiba
"""
import onnx
import onnxruntime as ort
import numpy as np

# model = onnx.load('../weight/mobilenet_v2-b0353104.onnx')
# onnx.checker.check_model(model)

# ort_session = ort.InferenceSession("../weight/MobileNetV2_level1_newest.onnx", providers=['CUDAExecutionProvider'])
ort_session = ort.InferenceSession(r"F:\model_pretrain\resnet18-f37072fd.onnx", providers=['CUDAExecutionProvider'])

outputs = ort_session.run(
    None,
    {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)},
)
print(outputs)
print(outputs[0].shape)
