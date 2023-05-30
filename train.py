from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import torchmetrics
# from model.unet_model_ln.unet_model_ln import UNet
# from model.unet_model_seattention import UNet
# from model.unet_model_mobilevit import UNet
# from model.unet_model import UNet
# from model.shift_unet.unet_model import UNet
from model.biformer_unet.unet_model import UNet
# from metrics import SegmentationMetric
from utils.metrics import SegmentationMetric

def train_net(net, device, data_path, epochs = 60, batch_size=2, lr = 0.00001, file_type='png'):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path, file_type)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义损失函数RMSprop
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法，使用指标为Binary Cross Entropy并且计算前先使用sigmoid函数归一化
    criterion = nn.BCEWithLogitsLoss() 

    # # 计算评价指标并记录---
    # metric = SegmentationMetric(2) # 2为类别数

    # best_loss 统计，初始化为正无穷
    best_loss = float('inf')
    record = time.gmtime(time.time() + 8*60*60)
    # 将record转换为字符串格式为: 2021-04-22T06:00:22
    record = time.strftime("%Y-%m-%dT%H_%M_%S", record)
    record = record + input("请输入备注信息:")
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

            ## 计算评价指标并记录---
            ## 将预测出的二值化图像记录
            # pred_label = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred)) # 阈值为0.5
            # metric.addBatch(pred_label, label)
            # pa = metric.pixelAccuracy()
            # writer.add_scalar(f'pa{record}/train' , pa, epoch)
            # cpa = metric.classPixelAccuracy()
            # writer.add_scalar(f'cpa{record}/train' , cpa, epoch)
            # mpa = metric.meanPixelAccuracy()
            # writer.add_scalar(f'mpa{record}/train' , mpa, epoch)
            # IoU = metric.intersectionOverUnion()
            # writer.add_scalar(f'IoU{record}/train' , IoU, epoch)
            # mIoU = metric.meanIntersectionOverUnion()
            # writer.add_scalar(f'mIoU{record}/train' , mIoU, epoch)
        
        metric_dict = metric.compute()   
        metric.reset()
        writer.add_scalar(f'train{record}/precision' , metric_dict["pre"], epoch)
        writer.add_scalar(f'train{record}/recall' ,metric_dict['rec'], epoch)
        writer.add_scalar(f'train{record}/accuracy' , metric_dict['acc'], epoch)
        writer.add_scalar(f'train{record}/f1' , metric_dict['f1'], epoch)
        writer.add_scalar(f'train{record}/dice' , metric_dict['dice'], epoch)
        writer.add_scalar(f'train{record}/hd' , metric_dict['hd'], epoch)


    print("best_loss: ", best_loss)
    log = open('log' + record + '.txt', 'w')
    # log.write("epoches: " + epoch + "\n"  + "batch_size:" + batch_size + "\n" + "lr:" + lr + "\n" + "best_loss:" + best_loss + "\n")
    log.write(f"epochs: {epochs}\nbatch_size: {batch_size}\nlr: {lr}\nbest_loss: {best_loss}\n")
    log.close()
    torch.save(net.state_dict(), 'best_model' + record + '.pth')

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 加载网络，图片单通道1，分类为1
    # net = UNet(n_channels=3, n_classes=1)
    net = UNet(n_channels=1, n_classes=1).to(device=device)

    # 添加tensorboard
    writer = SummaryWriter(log_dir='logs', comment='UNet')

    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    # data_path = "xirou/train/"
    # data_path = "shengnong/train"
    data_path = "new_data/train"

    train_net(net, device, data_path, 60, 16, file_type='png')
    writer.close()