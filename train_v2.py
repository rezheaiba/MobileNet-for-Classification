import datetime
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm
from torchvision.models import *
from model_v3 import MobileNetV3


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    # transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = r'../'  # get data root path
    image_path = os.path.join(data_root, "final-450-gray")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    class_list = train_dataset.class_to_idx  # 类别和index的映射，这是ImageFolder处理后的一个参数
    cla_dict = dict((val, key) for key, val in class_list.items())  # 映射存为字典格式
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)  # 把数据转json文件，indent：数据格式缩进显示4个空格，读起来更加清晰
    with open('final_class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 80
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # nw = 0
    print('Using {} dataloader workers every process'.format(nw))

    writer = SummaryWriter('final')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # 需要先用六分类的网络去加载之前训练出来的网络参数
    # net = MobileNetV2(6)
    # model_weight_path = 'weight-init/MobileNetV2_level1_best.pth'
    # # 把训练好的权重加载到网络上，其中torch.load先把字典型权重读取出来，然后net再把读取的参数加载
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # # change fc layer structure
    # in_channel = net.classifier[1].in_features
    # net.classifier = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(in_channel, 7)
    # )

    net = MobileNetV3(dilated=True, num_classes=7, arch='mobilenet_v3_small',reduced_tail=True)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.0).to(device)

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2,gamma=0.1)

    epochi = 0
    epochs = 51
    best_acc = 0.0
    # best_acc = 0.98
    save_path_best = 'weight-final-v2/mobilenetV2_best.pth'
    save_path_newest = 'weight-final-v2/mobilenetV2_newest.pth'
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    prev_time = time.time()

    if epochi != 0:
        net.load_state_dict(torch.load(save_path_newest, map_location=device))
    for epoch in range(epochi, epochs):
        # train
        net.train()
        running_loss = 0.0
        running_val_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            # image.shape = (4, 3, 224, 224) label.shape = (4,)->tensor([65, 52, 12, 44])
            images, labels = data
            optimizer.zero_grad()
            # logits.shape = (4,67)
            logits = net(images.to(device))
            # 自动把label变成one-hot
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # left time
            batches_done = epoch * train_steps + step  # n_epochs所遍历的总batches
            batches_left = epochs * train_steps - batches_done  # 总共剩余的batches
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()  # 某epoch中某batch的时刻

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} left-time:{}".format(epoch + 1,
                                                                                  epochs,
                                                                                  loss,
                                                                                  time_left)
            writer.add_scalar('train_loss', loss, epoch * train_steps + step)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for step, val_data in enumerate(val_bar):
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                # torch.max(input,dim)是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值,return:values(值),indices(索引)
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # torch.ep：逐元素比较是否相等返回TorF/1or0

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

                val_loss = loss_function(outputs, val_labels.to(device))
                writer.add_scalar('val_loss', val_loss, epoch * val_steps + step)
                running_val_loss += val_loss.item()

        val_accurate = acc / val_num
        # scheduler.step()
        print('[epoch %d] train_loss: %.3f val_loss: %.3f val_accuracy: %.3f lr: %f' %
              (epoch + 1, running_loss / train_steps, running_val_loss / val_steps, val_accurate,
               optimizer.param_groups[0]['lr']))
        # 保存最新一次的权重
        torch.save(net.state_dict(), save_path_newest)

        # 保存准确率最高的权重
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path_best)

    print('Finished Training')


if __name__ == '__main__':
    main()