import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model_v2 import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = r"C:\Users\A\Pictures\IQIYISnapShot"
    imgs_root = r"D:\Dataset\data\test_dataset\水波纹"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = 'final_class_indices_8.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = MobileNetV2(8).to(device)

    # load model weights
    weights_path = "weight-final-8-1/MobileNetV2_final_best.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 1  # 每次预测时将多少张图片打包成一个batch
    count = 0
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            multi = torch.tensor(100, dtype=torch.float)
            predict = predict * multi
            predict = predict.type(torch.int64)
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                # if class_indict[str(cla.numpy())] != 'fog' or pro.numpy()<0.9:
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                     class_indict[str(cla.numpy())],
                                                                     pro.numpy()))
                print(predict)

            if class_indict[str(cla.numpy())] == 'stripe':  # and pro.numpy() > 0.8:
                # if class_indict[str(cla.numpy())] == 'stronglight':
                count += 1
        print("平均准确率：{:.3}".format(count / len(img_path_list)))


if __name__ == '__main__':
    main()
