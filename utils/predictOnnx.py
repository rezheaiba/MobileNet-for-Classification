"""
# @Time    : 2022/8/8 18:40
# @File    : predictOnnx.py
# @Author  : rezheaiba
"""
import torch, cv2
import numpy as np
import torch.nn.functional as F
from torchvision import models
from model_v2 import MobileNetV2
import matplotlib.pyplot as plt
import os
import onnxruntime as ort

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def preict_one_img(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 把图片BGR变成RGB
    print(img.shape)

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32)
    img /= 255
    print(img.shape)

    outputs = ort_session.run(
        None,
        {"input": img.astype(np.float32)},
    )
    print(np.max(outputs[0]))
    print(np.argmax(outputs[0]))

    out = torch.tensor(outputs[0], dtype=torch.float64)
    out = F.softmax(out, dim=1)
    proba, class_id = torch.max(out, 1)

    proba = float(proba[0])
    class_id = int(class_id)
    img = img.squeeze(0)
    new_img = np.transpose(img, (1, 2, 0))
    plt.imshow(new_img)
    plt.title("predicted class: %s .  probability: %3f" % (classes[class_id], proba))
    plt.show()


if __name__ == '__main__':
    classes = MobileNetV2(4).IMAGENET1K_V1.value.meta["categories"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = "../images/8.jpg"
    model_path = "../weight/MobileNetV2_level1_newest.onnx"
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    preict_one_img(img_path)
