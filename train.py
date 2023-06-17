import argparse
import logging
import os
import random
import warnings
from pydoc import locate

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import model
from model.doubleunet_pytorch import build_doubleunet

from trainer import trainer_synapse
warnings.filterwarnings("ignore")

# TODO
attention_enum_dict = {0:"none", 1:"mobile_vit", 2:"biformer"}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="../data/Synapse/train_npz",
    help="root dir for train data",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="../data/Synapse/test_vol_h5",
    help="root dir for test1 data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
# parser.add_argument("--max_iterations", type=int, default=200, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=1, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size per gpu")
parser.add_argument("--num_workers", type=int, default=0, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=1, help="eval_interval")
parser.add_argument("--model_name", type=str, default="synapse", help="model_name")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network base learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--z_spacing", type=int, default=1, help="z_spacing")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")
parser.add_argument(
    "--module",
    # default=networks.doubleunet.DoubleUNet,
    default=model.doubleunet_pytorch.build_doubleunet,
    help="The module that you want to load as the network, e.g. model.doubleunet_pytorch.build_doubleunet",
)
parser.add_argument("--in_channels", type=int, default=1, help="the channel of input image, default is 1")
parser.add_argument("--patch_size", type=int, default=4, help="the patch size of input image, default is 4")
parser.add_argument("--has_se", type=int, default=1, help="whether to use se module in each conv block, default is True")
parser.add_argument("--attentions", type=str, default="00", help=f"the attention in this doubleunet one for unet1, another one for unet2, default is 00, you can choose from {attention_enum_dict}, exsample: `python train.py --attentions 12`")

args = parser.parse_args()


if __name__ == "__main__":
    # setting device on GPU if available, else CPU，选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()



    import ast

    assert len(args.attentions) == 2, "the length of attentions must be 2, [0] for unet1, [1] for unet2"
    for i in args.attentions:
        assert int(i) in attention_enum_dict.keys(), f"the attentions must be in {attention_enum_dict.keys()}"
    args.attentions = [attention_enum_dict[int(i)] for i in args.attentions]
    args.has_se = bool(args.has_se)

    print(args) # Additionast.literal_eval(str(args.attentions))al Info when using cuda，检查设备，尝试分配内存，并检查内存使用情况

    
    # 测试显存分配
    net = build_doubleunet(in_channels=args.in_channels, num_classes=args.num_classes, patch_size=args.patch_size, attentions=args.attentions).cuda(0)
    if device.type == "cuda":
        print("the cuda used is"+torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"



    # 似乎涉及启动cudnn加速，但是mobilevit不支持cudnn加速算法
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # 随机参数固定化
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 9,
            "in_channels": 1,
        },
    }
    print(args.root_path)
    print(args.list_dir)
    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    print(args.base_lr)

    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.in_channels = dataset_config[dataset_name]["in_channels"]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # net = transformer(num_classes=args.num_classes).cuda(0)
    net = build_doubleunet(in_channels=args.in_channels, num_classes=args.num_classes, patch_size=args.patch_size, attentions=args.attentions, has_se=args.has_se).cuda(0)
    trainer = {
        "Synapse": trainer_synapse,
    }
    print(trainer[dataset_name])
    trainer[dataset_name](args, net, args.output_dir)
