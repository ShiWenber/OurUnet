import argparse
import glob
import numpy as np
import torch
import os
import cv2
from torch.utils.tensorboard import SummaryWriter
# from model.unet_model import UNet
# from model.unet_model_seattention import UNet
# from model.unet_model_mobilevit import UNet
from model.biformer_unet.unet_model import UNet
from utils.metrics import SegmentationMetric
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--data_test_path', type=str, default='new_data/test', help='root dir of data')
    parser.add_argument('--model_path', type=str, default='best_model2023-05-31T09_47_52.pth', help='model path')
    parser.add_argument('--save_path', type=str, default='temp_res', help='save path')

    args = parser.parse_args()

    # 判断 save_path 是否存在,不存在则创建
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if save_path[-1] != '/':
        save_path = save_path + '/'


    start_time = time.time()
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络,图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 载入到device中
    net.to(device=device)
    # 加载参数
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    # 测试
    net.eval()

    # tensorboard
    # writer = SummaryWriter(log_dir='logs', comment='UNet')

    # 读取图片路径
    tests_path = glob.glob(args.data_test_path + r'/image/*')
    # 遍历图片
    print("tests_path:", tests_path)
    for test_path in tests_path:
        # 保存结果图片
        file_name = os.path.basename(test_path)
        print("file_name:", file_name)
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
        cv2.imwrite(save_path + file_name, pred)

    print("time:", time.time() - start_time)
