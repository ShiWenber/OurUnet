import glob
import numpy as np
import torch
import os
import cv2
from torch.utils.tensorboard import SummaryWriter
# from model.unet_model import UNet
# from model.unet_model_seattention import UNet
from model.unet_model_mobilevit import UNet
from utils.metrics import SegmentationMetric
import time

if __name__ == "__main__":

    start_time = time.time()
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络,图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 载入到device中
    net.to(device=device)
    # 加载参数
    net.load_state_dict(torch.load('best_model2023-01-14T01_54_20shengnong1400_unetamvit_60_4_10.pth', map_location=device))
    # 测试
    net.eval()

    # tensorboard
    # writer = SummaryWriter(log_dir='logs', comment='UNet')

    # 读取图片路径
    tests_path = glob.glob("shengnong/valid/*")
    # 遍历图片
    print("tests_path:", tests_path)
    for test_path in tests_path:
        # 保存结果图片
        save_res_path = test_path.split(".")[0] + "_res.png"
        # 读取
        img = cv2.imread(test_path)
        # 转灰度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 转为batch为1,通道为1的大小为512*512的数组
        img = img.reshape(1,1,img.shape[0], img.shape[1])
        # img = img.reshape(1,img.shape[0], img.shape[1])
        # 转tensor
        img_tensor = torch.from_numpy(img)
        # 转到device中
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)

        # tensorboard记录中间特征图
        # writer.add_graph(net, img_tensor)
        # writer.add_image('input', img_tensor)
        # writer.add_image('output', pred)
        

        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 计算acc


        # 保存图片
        print(save_res_path)
        cv2.imwrite(save_res_path, pred)

    print("time:", time.time() - start_time)
