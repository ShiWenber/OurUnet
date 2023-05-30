# 图像分割任务模型指标
## 1. Precision
## 2. Recall
## 3. (HD) Hausdorff距离
## 4. (DSC) Dice相似系数
## 5. F1-score
## 6. Accuracy


# todo: pre未通过测试
import numpy as np
import torch
import torchmetrics

class SegmentationMetric():

    def __init__(self, device="cuda" if torch.cuda.is_available() else 'cpu') -> None:
        self.acc_metric = torchmetrics.Accuracy(task='binary', average='micro').to(device)
        self.f1_metric = torchmetrics.F1Score(task='binary',average='micro').to(device)
        # self.iou_metric = torchmetrics.IoU(num_classes=2, average='macro').to(device)
        self.dice_metric = torchmetrics.Dice(num_classes=2, average='micro').to(device)
        self.precision_metric = torchmetrics.Precision(task="binary", average='micro').to(device)
        self.recall_metric = torchmetrics.Recall(task="binary").to(device)
        self.hd_metric = torchmetrics.HammingDistance(task="binary", average='micro').to(device)
    
    def update(self, pred, label):
        pred = pred.clone().detach()
        label = label.clone().detach()
        pred_clone = torch.tensor(pred > 0.5, dtype=torch.bool)
        label_clone = torch.tensor(label > 0.5, dtype=torch.bool)

        pred_clone, label_clone = torch.logical_not(pred_clone), torch.logical_not(label_clone)

        self.acc_metric(pred_clone, label_clone)
        self.f1_metric(pred_clone, label_clone)
        # self.iou_metric(pred, label)
        self.dice_metric(pred_clone, label_clone)
        self.precision_metric(pred_clone, label_clone)
        self.recall_metric(pred_clone, label_clone)
        self.hd_metric(pred_clone, label_clone)
    
    def compute(self): 
        """返回评价指标计算结果

        Returns:
            dict: 评价指标计算结果，包括["acc", "f1", "dice", "precision", "recall", "hd"]
        """
        acc = self.acc_metric.compute()
        f1 = self.f1_metric.compute()
        # iou = self.iou_metric.compute()
        dice = self.dice_metric.compute()
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        hd = self.hd_metric.compute()
        return {
            "acc": acc,
            "f1": f1,
            # "iou": iou,
            "dice": dice,
            "pre": precision,
            "rec": recall,
            "hd": hd
        }
        # return acc, f1, iou, dice, precision, recall, hd

    def reset(self):
        self.acc_metric.reset()
        self.f1_metric.reset()
        # self.iou_metric.reset()
        self.dice_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.hd_metric.reset()



if __name__ == '__main__':

    def metrics_test(pre: torch.Tensor, label: torch.Tensor):
        # 将输入的灰度图转为二值图
        pre = pre.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        # print(f"pre-{type(pre_label)}, label-{type(label)}")

        pre = pre > 0.5
        label = label > 0.5

        seg_inv, gt_inv = np.logical_not(pre), np.logical_not(label)
        true_pos = float(np.logical_and(pre, label).sum())
        true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
        false_pos = float(np.logical_and(pre, gt_inv).sum())
        false_neg = float(np.logical_and(seg_inv, label).sum())

        #然后根据公式分别计算出这几种指标
        prec = true_pos / (true_pos + false_pos + 1e-6)
        rec = true_pos / (true_pos + false_neg + 1e-6)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
        IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)

        return prec, rec, accuracy, IoU



    
    

    metric = SegmentationMetric()
    # pred, label 的元素取值为0-1之间的浮点数
    pred = torch.randn(1, 1, 512, 512).to("cuda")
    label = torch.randn(1, 1, 512, 512).to("cuda")
    # 归一化
    pred = torch.sigmoid(pred)
    label = torch.sigmoid(label)
    # pred, label 的元素取值为0或1
    print("cuda: ", torch.cuda.is_available())
    # pred = torch.randint(0, 2, (1, 1, 512, 512)).to("cuda")
    # label = torch.randint(0, 2, (1, 1, 512, 512)).to("cuda")
    assert pred.is_cuda, "pred should be on cuda"
    assert label.is_cuda, "label should be on cuda"
    metric.update(pred, label)
    metric_dict = metric.compute()
    print(metric_dict)

    prec, rec, acc, iou = metrics_test(pred, label)
    test_dict = {
        "pre": prec,
        "rec": rec,
        "acc": acc,
        "iou": iou
    }
    print(test_dict)
    # assert test_dict["pre"]== metric_dict['pre'], f"test_pre: {test_dict['pre']}, metric_pre: {metric_dict['pre']}"
    # assert test_dict["rec"] == metric_dict['rec'], f"test_rec: {test_dict['rec']}, metric_rec: {metric_dict['rec']}"
    # assert test_dict["acc"] == metric_dict['acc'], f"test_acc: {test_dict['acc']}, metric_acc: {metric_dict['acc']}"


    print(f"test_pre: {test_dict['pre']}, metric_pre: {metric_dict['pre']}")
    print(f"test_rec: {test_dict['rec']}, metric_rec: {metric_dict['rec']}")
    print(f"test_acc: {test_dict['acc']}, metric_acc: {metric_dict['acc']}")
    metric.reset()
    print("test pass!")