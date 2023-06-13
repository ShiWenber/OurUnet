import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import DiceLoss
from torchvision import transforms
from utils.utils import test_single_volume
from torch.nn import functional as F
# from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from datasets.dataset_synapse import Synapse_dataset
import matplotlib.pyplot as plt
import pandas as pd
import datetime


def inference(model, testloader, args, test_save_path=None):
    # 将模型设置为评估模式。
    model.eval()
    # 初始化度量列表，用于累加各批次的度量值。
    metric_list = 0.0
    # 遍历测试数据加载器中的批次。
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # 获取图像的高度和宽度。
        h, w = sampled_batch["image"].size()[2:]
        # 从当前批次中提取图像、标签和案例名称。
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # 对单个体素进行测试。
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        # 将当前批次的度量值累加到度量列表中。
        metric_list += np.array(metric_i)
        # 打印日志，显示当前批次的度量值。
        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    # 计算整个测试集的平均度量值。
    metric_list = metric_list / len(testloader.dataset)
    # 计算整个测试集的平均度量值。
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    # 计算总体性能指标。
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    # 打印测试性能。
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    # 打印测试性能。
    return performance, mean_hd95


def plot_result(dice, h, snapshot_path,args):
    # 创建一个字典，包含平均Dice系数和平均Hausdorff距离。
    dict = {'mean_dice': dice, 'mean_hd95': h}
    # 创建一个字典，包含平均Dice系数和平均Hausdorff距离。
    df = pd.DataFrame(dict)
    # 创建一个新的图形（索引为0）。
    plt.figure(0)
    # 绘制平均Dice系数曲线。
    df['mean_dice'].plot()
    # 设置图形分辨率。
    resolution_value = 1200
    # 设置图形标题。
    plt.title('Mean Dice')
    # 获取当前日期和时间。
    date_and_time = datetime.datetime.now()
    # 生成文件名，包含模型名称、日期和时间。
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    # 创建要保存文件的路径。
    save_mode_path = os.path.join(snapshot_path, filename)
    # 以指定的分辨率保存图形为PNG文件。
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    # 创建一个新的图形（索引为1）。
    plt.figure(1)
    # 绘制平均Hausdorff距离曲线。
    df['mean_hd95'].plot()
    # 设置图形标题。
    plt.title('Mean hd95')
    # 为Hausdorff距离生成文件名。
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    # 创建要保存文件的路径。
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv
    # 为CSV文件生成文件名。
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    # 创建要保存文件的路径。
    save_mode_path = os.path.join(snapshot_path, filename)
    # 将DataFrame保存为CSV文件，使用制表符分隔。
    df.to_csv(save_mode_path, sep='\t')


def trainer_synapse(args, model, snapshot_path):
    # 创建存储测试结果的文件夹。
    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')
    # 设置日志记录。
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # 设置基本训练参数。
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # 定义数据预处理操作。
    x_transforms = transforms.Compose([
        # 表示将数据类型转换为张量。
        transforms.ToTensor(),
        # 表示将张量的值按照均值0.5和标准差0.5进行归一化。
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()
    # 创建训练数据集和数据加载器。
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=args.img_size,
                               norm_x_transform=x_transforms, norm_y_transform=y_transforms)

    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
    #                          worker_init_fn=worker_init_fn)

    # 保证取数据随机，内存不够大不要选pin_memory为True
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False,
                             worker_init_fn=worker_init_fn)
    # 创建测试数据集和数据加载器。
    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir, img_size=args.img_size)

    # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 保证测试集固定，防止结果波动
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    # 如果有多个GPU，则使用DataParallel。
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    # 设置模型为训练模式。
    model.train()
    # 定义损失函数和优化器。
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    print("snapshot_path", snapshot_path)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    writer = SummaryWriter(snapshot_path + '/log' + time_str)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    # 初始化性能指标。
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    dice_ = []
    hd95_ = []
    # 开始训练循环。
    for epoch_num in iterator:
        # 遍历批次。
        for i_batch, sampled_batch in enumerate(trainloader):
            # 获取图像和标签批次。
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print("data shape---------", image_batch.shape, label_batch.shape)
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            outputs = model(image_batch)
            # outputs = F.interpolate(outputs, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
            # 计算损失。
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            # print("loss-----------", loss)
            # 反向传播和优化。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新学习率。
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # 更新迭代次数。
            iter_num = iter_num + 1
            # 将训练信息写入TensorBoard。
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            # 记录训练信息。
            # logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            # 可视化训练结果。
            # if iter_num % 20 == 0:
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                # writer.add_image('train/Image', image, iter_num / batch_size)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num / batch_size)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                # writer.add_image('train/GroundTruth', labs, iter_num / batch_size)
        # Test
        # 测试
        eval_interval = args.eval_interval 
        if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % eval_interval == 0:
            # 保存模型。
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # 运行推理。
            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            print(f"Epoch {epoch_num}")
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()
        # 当达到最大迭代次数时，保存模型并退出循环。
        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            if not (epoch_num + 1) % args.eval_interval == 0:
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
                print(f"Epoch {epoch_num}, Last Epcoh")
                mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()
                
            iterator.close()
            break
    # 绘制结果并关闭资源。
    plot_result(dice_, hd95_, snapshot_path, args)
    writer.close()
    return "Training Finished!"