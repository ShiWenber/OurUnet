# dataset.py
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class MyLoader(Dataset):
    def __init__(self, data_path, file_type='png'):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        # glob.glob()返回所有匹配的文件路径列表(减化的正则表达式匹配)
        self.imgs_path = glob.glob(os.path.join(data_path, r'image/*.' + file_type))
    def augment(self, image, flipCode):
        # 使用cv2.flip做数据增强，flipCode为1表示水平翻转，0表示垂直翻转，-1表示水平垂直翻转
        # flip = cv2.flip(image, flipCode) # todo
        flip = image # todo
        return flip
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签 
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转化为单通道的灰度图
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # todo 
        # print(f"shape image, label:{image.shape}  {label.shape}")

        # reshape为 3 * 288 * 384 的输入形式
        image = image.transpose(2, 0, 1)

        # label reshape为 1 * 288 * 384 的输入形式
        label = label.reshape(1, label.shape[0], label.shape[1])

        # label = label.transpose(2, 0, 1)
        # image = image.reshape(1, image.shape[0], image.shape[1])
        # label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，如果标签像素值超过1 将标签像素值映射到0-1之间
        # print(f"shape image, label:{image.shape}  {label.shape}")
        if label.max() > 1:
            label = label / 255
        # 随机数据增强
        flipCode  = random.choice([-1, 0, 1, 2])
        # 2表示不做处理
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label
    def __len__(self):
        # 返回数据集的大小
        return len(self.imgs_path)

if __name__ == "__main__":
    train_dataset = MyLoader('../data/xirou_rgb/train')
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)