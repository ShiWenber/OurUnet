# 实验记录

## Q & A

if you get the error like this:

```bash
Traceback (most recent call last):
  File "train.py", line 144, in <module>
    trainer[dataset_name](args, net, args.output_dir)
  File "/root/models/unet/trainer.py", line 179, in trainer_synapse
    loss.backward()
  File "/opt/conda/lib/python3.7/site-packages/torch/_tensor.py", line 396, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "/opt/conda/lib/python3.7/site-packages/torch/autograd/__init__.py", line 175, in backward
    allow_unreachable=True, accumulate_grad=True)  # Calls into the C++ engine to run the backward pass
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
```

you may think about it is the problem of mobile_vit or cuDNN, but it is not. It is a cuda memory problem. You can try to reduce the batch size.

## train command

```bash
python train.py --num_workers 0 --max_epochs 50 --batch_size 12 --n_gpu 1 --patch_size 4  --has_se True --attentions 00
```

`has_se` means whether to use SE module, `attentions` means the attentions used in model（first num for unet1, second num for unet2）

<!-- 目前有 mobile_vit 和 biformer -->
So far, supported attentions is

```python
attention_enum_dict = {0:"none", 1:"mobile_vit", 2:"biformer"}
```