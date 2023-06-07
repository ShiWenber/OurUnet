import argparse
import glob
import json

import cv2
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import time
# from model.unet_model_ln.unet_model_ln import UNet
# from model.unet_model_seattention import UNet
# from model.unet_model_mobilevit import UNet
# from model.unet_model import UNet
# from model.shift_unet.unet_model import UNet
# from model.biformer_unet.unet_model import UNet
# from metrics import SegmentationMetric
from utils.metrics import SegmentationMetric

def train_net(net, device, data_path, record, epochs = 60, batch_size=2, lr = 0.00001, file_type='png'):
    # 加载训练集
    train_dataset = ISBI_Loader(data_path + "/train", file_type)
    test_dataset = ISBI_Loader(data_path + "/test", file_type)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              num_workers=2,
                                              shuffle=False)
    # 定义损失函数RMSprop
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法，使用指标为Binary Cross Entropy并且计算前先使用sigmoid函数归一化
    criterion = nn.BCEWithLogitsLoss() 

    # # 计算评价指标并记录---
    # metric = SegmentationMetric(2) # 2为类别数

    # best_loss 统计，初始化为正无穷
    best_loss = float('inf')
    
    print(record)
    metric = SegmentationMetric()
    # 训练epochs次
    for epoch in range(epochs):
        net.train()
        # accumulated_loss = tensor.Places365Tensor(0)
        accumulated_loss = 0
        count = 0
        # 按照batch_size进行训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 数据拷贝到GPU
            image_g = image.to(device=device, dtype=torch.float32)
            label_g = label.to(device=device, dtype=torch.float32)
            # 使用网络参数预测
            pred = net(image_g)
            # print("pred: ", pred)
            # print("label_g: ", label_g.max())
            assert pred.is_cuda, "pred is not cuda"
            # 计算loss
            loss = criterion(pred, label_g)

            accumulated_loss += loss.item()            
            count += 1

            metric.update(pred, label_g)



            # # tensorboard中显示特征图
            # # writer.add_graph(net, image)
            # # 显示中间结果(仅仅显示效果最好的)
            # features = net.features
            # for i in features.keys():
            #     writer.add_image(f"features/{i}" + record, features[i][0][0].detach().cpu().unsqueeze(dim=0), epoch)


            # 保存loss最小的模型
            if loss < best_loss:
                best_loss = loss
                # tensorboard中显示特征图
                # writer.add_graph(net, image)
                
            # 更新参数
            loss.backward() # TODO 反向传播
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, accumulated_loss / count))
        # # 显示中间结果(仅仅显示效果最好的)并保存日志
        # features = net.features
        # for i in features.keys():
        #     writer.add_image(f"features{record}/{i}", features[i][0][0].detach().cpu().unsqueeze(dim=0), epoch)

        writer.add_scalar(f'train{record}/loss' , loss.item(), epoch)
        
        metric_dict = metric.compute()   
        metric.reset()
        writer.add_scalar(f'train{record}/precision' , metric_dict["pre"], epoch)
        writer.add_scalar(f'train{record}/recall' ,metric_dict['rec'], epoch)
        writer.add_scalar(f'train{record}/accuracy' , metric_dict['acc'], epoch)
        writer.add_scalar(f'train{record}/f1' , metric_dict['f1'], epoch)
        writer.add_scalar(f'train{record}/dice' , metric_dict['dice'], epoch)
        writer.add_scalar(f'train{record}/hd:' , metric_dict['hd'], epoch)
        print("train_acc", metric_dict["acc"])

        # 测试--
        net.eval()
        # 读取图片路径
        for img, label in test_loader:
            img = img.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(img)
            metric.update(pred, label)

        metric_dict = metric.compute()
        metric.reset()
        writer.add_scalar(f'train{record}/accuracy' , metric_dict["acc"], epoch)
        print("test_acc", metric_dict["acc"])



    print("best_loss: ", best_loss)
    log = open('log' + record + '.txt', 'w')
    # log.write("epoches: " + epoch + "\n"  + "batch_size:" + batch_size + "\n" + "lr:" + lr + "\n" + "best_loss:" + best_loss + "\n")
    log.write(f"epochs: {epochs}\nbatch_size: {batch_size}\nlr: {lr}\nbest_loss: {best_loss}\n")
    log.close()
    torch.save(net.state_dict(), 'best_model' + record + '.pth')
    return net

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ourunet")
    # parser.add_argument("--dataname", type=str, default="ENZYMES")
    parser.add_argument("--data_path", type=str, default="xirou")
    parser.add_argument("--data_file_type", type=str, default="png")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--n_classes", type=int, default=1)
    parser.add_argument("--n_channels", type=int, default=1)


    args = parser.parse_args()

    print(args)

    # 将命令行参数转换为字典类型
    hparams_dict = vars(args)
    # 将字典转换为 JSON 格式的字符串
    hparams_json_str = json.dumps(hparams_dict)

    # from model.unet_model_ln.unet_model_ln import UNet
    if args.model == "unet":
        from model.unet_model import UNet
    elif args.model == "unet_ln":
        from model.unet_model_ln.unet_model_ln import UNet    
    elif args.model == "unet_se":
        from model.unet_model_seattention import UNet
    elif args.model == "unet_mobilevit":
        from model.unet_model_mobilevit import UNet
    elif args.model == "unet_shift":
        from model.shift_unet.unet_model import UNet
    elif args.model == "unet_biformer":
        from model.biformer_unet.unet_model import UNet
    elif args.model == "unet_mobilevit_biformer":
        from model.mvit_biformer_unet.unet_model import UNet
    else:
        raise ValueError("model name error")
    
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 加载网络，图片单通道1，分类为1
    # net = UNet(n_channels=3, n_classes=1)
    net = UNet(n_channels=args.n_channels, n_classes=args.n_classes).to(device=device)

    record = time.gmtime(time.time() + 8*60*60)
    # 将record转换为字符串格式为: 2021-04-22T06:00:22
    record = time.strftime("%Y-%m-%dT%H_%M_%S", record)
    # 添加tensorboard
    writer = SummaryWriter(log_dir='logs', comment='UNet')

    writer.add_text(f'train{record}/hparams', hparams_json_str)
   
        

    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    # data_path = "xirou/train/"
    # data_path = "shengnong/train"
    # data_path = "new_data/train"

    net = train_net(net, device, args.data_path, record,  args.epochs, args.batch_size, file_type=args.data_file_type, lr=args.lr)


    # 测试
    # metric = SegmentationMetric()

    

    writer.close()