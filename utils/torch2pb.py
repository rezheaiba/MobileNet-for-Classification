"""
# @Time    : 2022/8/9 17:08
# @File    : torch2pb.py
# @Author  : rezheaiba
"""
# onnx-tf要求tensorflow==2.2.0
# onnx-tf==1.6

import onnx
from onnx_tf.backend import prepare


#
onnx_model = onnx.load("../weight/MobileNetV2_level1_best.onnx")  # load onnx model
tf_exp = prepare(onnx_model)  # prepare tf representation
tf_exp.export_graph("../weight/MobileNetV2_level1_best.pb")  # export the model