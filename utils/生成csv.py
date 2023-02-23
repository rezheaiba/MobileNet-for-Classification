"""
# @Time    : 2023/2/7 11:17
# @File    : 生成csv.py
# @Author  : rezheaiba
"""
import csv
import os

os.makedirs('label', exist_ok=True)
f_train = open('label/train_label.csv', 'w', encoding='utf-8', newline='')
f_test = open('label/test_label.csv', 'w', encoding='utf-8', newline='')
csv_writer_train = csv.writer(f_train)
csv_writer_test = csv.writer(f_test)

root = r'D:\Dataset\data\MTest'

if __name__ == '__main__':
    # 构建表头
    # fileHeader = ["path", "label1", "label2", "label3", "label4"]
    # csv_writer_train.writerow(fileHeader)
    # csv_writer_test.writerow(fileHeader)

    # get the path of the train_dataset
    '''二全连接'''
    for h, d, f in os.walk(os.path.join(root, 'train')):
        os.chdir(h)
        for trainName in f:
            print(trainName)
            # label1:0-indoor 1-outdoor
            # label2:0-flying-dust 1-fog 2-rainy 3-wdr 4-stronglight 5-stripe 6-other
            # label3:0-stronglight 1-other
            # label4:0-stripe 1-other
            if 'indoor' in trainName:
                label1, label2 = '0', '6'
            if 'outdoor' in trainName:
                label1, label2 = '1', '6'
            if 'flying-dust' in trainName:
                label1, label2 = '1', '0'
            if 'fog' in trainName:
                label1, label2 = '1', '1'
            if 'rainy' in trainName:
                label1, label2 = '1', '2'
            if 'wdr' in trainName:
                label1, label2 = '0', '3'
            if 'stronglight' in trainName:
                label1, label2 = '1', '4'
            if 'stripe' in trainName:
                label1, label2 = '0', '5'
            csv_writer_train.writerow([trainName, label1, label2])
    # get the path of the test_dataset
    for h, d, f in os.walk(os.path.join(root, 'test')):
        os.chdir(h)
        for testName in f:
            if 'indoor' in testName:
                label1, label2 = '0', '6'
            if 'outdoor' in testName:
                label1, label2 = '1', '6'
            if 'flying-dust' in testName:
                label1, label2 = '1', '0'
            if 'fog' in testName:
                label1, label2 = '1', '1'
            if 'rainy' in testName:
                label1, label2 = '1', '2'
            if 'wdr' in testName:
                label1, label2 = '0', '3'
            if 'stronglight' in testName:
                label1, label2 = '1', '4'
            if 'stripe' in testName:
                label1, label2 = '0', '5'
            csv_writer_test.writerow([testName, label1, label2])
    '''四全连接'''
    # for h, d, f in os.walk(os.path.join(root, 'train')):
    #     os.chdir(h)
    #     for trainName in f:
    #         print(trainName)
    #         # label1:0-indoor 1-outdoor
    #         # label2:0-flying-dust 1-fog 2-rainy 3-wdr 4-other
    #         # label3:0-stronglight 1-other
    #         # label4:0-stripe 1-other
    #         if 'indoor' in trainName:
    #             label1, label2, label3, label4 = '0', '4', '1', '1'
    #         if 'outdoor' in trainName:
    #             label1, label2, label3, label4 = '1', '4', '1', '1'
    #         if 'flying-dust' in trainName:
    #             label1, label2, label3, label4 = '1', '0', '1', '1'
    #         if 'fog' in trainName:
    #             label1, label2, label3, label4 = '1', '1', '1', '1'
    #         if 'rainy' in trainName:
    #             label1, label2, label3, label4 = '1', '2', '1', '1'
    #         if 'wdr' in trainName:
    #             label1, label2, label3, label4 = '1', '3', '1', '1'
    #         if 'stronglight' in trainName:
    #             label1, label2, label3, label4 = '1', '4', '0', '1'
    #         if 'stripe' in trainName:
    #             label1, label2, label3, label4 = '0', '4', '1', '0'
    #         csv_writer_train.writerow([trainName, label1, label2, label3, label4])
    # # get the path of the test_dataset
    # for h, d, f in os.walk(os.path.join(root, 'test')):
    #     os.chdir(h)
    #     for testName in f:
    #         if 'indoor' in testName:
    #             label1, label2, label3, label4 = '0', '4', '1', '1'
    #         if 'outdoor' in testName:
    #             label1, label2, label3, label4 = '1', '4', '1', '1'
    #         if 'flying-dust' in testName:
    #             label1, label2, label3, label4 = '1', '0', '1', '1'
    #         if 'fog' in testName:
    #             label1, label2, label3, label4 = '1', '1', '1', '1'
    #         if 'rainy' in testName:
    #             label1, label2, label3, label4 = '1', '2', '1', '1'
    #         if 'wdr' in testName:
    #             label1, label2, label3, label4 = '1', '3', '1', '1'
    #         if 'stronglight' in testName:
    #             label1, label2, label3, label4 = '1', '4', '0', '1'
    #         if 'stripe' in testName:
    #             label1, label2, label3, label4 = '0', '4', '1', '0'
    #         csv_writer_test.writerow([testName, label1, label2, label3, label4])
