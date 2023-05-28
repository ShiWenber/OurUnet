# 实验记录

- [ ] 用log实现自动化实验日志

数据集：原始30 + 锐化30
epochs: 60
batch_size: 1
best_loss: 0.0651

dataset: origin
epochs: 60
batch_size: 1
best_loss:  tensor(0.0849, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)

dataset: origin
epochs: 40
batch_size: 2
best_loss:  tensor(0.0967, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
end_loss: 0.1069

dataset: origin
epochs: 60
batch_size: 1
best_loss: best_loss:  tensor(0.0375, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
end_loss: 

best_loss:  tensor(0.0353, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)

dataset: 锐化30 + 原始30
epochs: 60
batch_size: 1
best_loss: best_loss:  tensor(0.0306, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)

dataset: 锐化30 + 原始30
epochs: 60
batch_size: 1
best_loss:  tensor(0.0353, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)

dataset: 原始30
epochs: 60
batch_size: 1
best_loss:  tensor(0.0523, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)

经过实验，60轮epoch下，锐化30 + 原始30的数据集loss能达到0.03左右，而原始30的数据集loss能达到0.05左右，使用注意力机制后，loss也能达到0.03左右