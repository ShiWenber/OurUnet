import os  # 引入 h5py 库，用于操作 HDF5 文件格式
import random
import h5py  # 引入 h5py 库，用于操作 HDF5 文件格式
import numpy as np
import torch
from scipy import ndimage  # 引入 SciPy 中的 ndimage 模块，用于图像处理
from scipy.ndimage.interpolation import zoom  # 引入 zoom 函数，用于改变图像大小
from torch.utils.data import Dataset
import imgaug as ia  # 引入 imgaug 库
import imgaug.augmenters as iaa  # 引入 imgaug 库的数据增强模块


def mask_to_onehot(mask, ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    # 将分割掩模（H，W，C）转换为（H，W，K），其中最后一个维度是一个one-hot编码向量，
    # C通常为1或3，K是类别数。
    semantic_map = []  # 初始化一个空列表，用于存储每个类别的二值掩模
    mask = np.expand_dims(mask, -1)  # 为 mask 添加一个维度，便于后续操作
    for colour in range(9):  # 对每个类别进行遍历
        # 对每个类别进行遍历
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)  # 将二值掩模添加到 semantic_map 列表中
    # 将所有二值掩模拼接成一个三维数组，最后一个维度是one-hot编码向量
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map


# 对分割掩模进行增强操作
def augment_seg(img_aug, img, seg):
    # 对分割掩模进行增强操作
    seg = mask_to_onehot(seg)
    # 使用 imgaug 库对图像进行增强
    aug_det = img_aug.to_deterministic() 
    image_aug = aug_det.augment_image(img)
    # 使用 imgaug 库对分割掩模进行增强
    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg)+1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug, segmap_aug


# 随机旋转和翻转操作
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


# 随机旋转操作
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


# 随机生成器类，用于对数据进行随机操作
class RandomGenerator(object):
    def __init__(self, output_size):
        # 设定输出大小
        self.output_size = output_size

    def __call__(self, sample):
        # 获取图像和标签
        image, label = sample['image'], sample['label']
        # 随机选择旋转翻转操作
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # 获取图像的尺寸
        x, y = image.shape
        # 若图像尺寸与设定的输出大小不一致，则对图像和标签进行缩放操作
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


# 自定义数据集类，继承自 PyTorch 的 Dataset 类
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, norm_x_transform=None, norm_y_transform=None):
        """
            分割数据集类
            :param base_dir: 数据根目录
            :param list_dir: 数据列表目录
            :param split: 数据集分割，可选值为 "train" 或 "test"
            :param img_size: 图像大小，用于调整图像大小
            :param norm_x_transform: 对输入图像进行归一化的变换，可选
            :param norm_y_transform: 对标签图像进行归一化的变换，可选
        """
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.img_size = img_size
        # 读取数据列表
        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),  # 随机垂直翻转
            iaa.Fliplr(0.5, name="Fliplr"),  # 随机水平翻转
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),  # 加入高斯噪音
            iaa.GaussianBlur(sigma=1.0),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 如果数据集是训练集。
        if self.split == "train":
            # 从样本列表中获取对应索引的切片名称。
            slice_name = self.sample_list[idx].strip('\n')
            # 构造数据路径。
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # 从文件中加载数据。
            data = np.load(data_path)
            # 获取图像和标签数据。
            image, label = data['image'], data['label']
            # 对图像和标签进行数据增强。
            image, label = augment_seg(self.img_aug, image, label)
            # 获取图像的尺寸。
            x, y = image.shape
            # 如果图像尺寸与预期尺寸不符，进行缩放操作。
            if x != self.img_size or y != self.img_size:
                # 使用三次样条插值进行缩放
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3)  # why not 3?
                # 使用最邻近插值进行缩放。
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)
        # 如果数据集是测试集。
        else:
            # 从样本列表中获取对应索引的卷名称。
            vol_name = self.sample_list[idx].strip('\n')
            # 构造文件路径。
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            # 从文件中加载数据。
            data = h5py.File(filepath)
            # 获取图像和标签数据。
            image, label = data['image'][:], data['label'][:]
        # 创建一个包含图像和标签的字典作为样本。
        sample = {'image': image, 'label': label}
        # print(f"shape image, label:{image.shape}  {label.shape}")
        # print(f"image, data:{image}  {data}")
        # 如果提供了图像的预处理操作，则将其应用于图像数据。(x 0.5中心归一化)
        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'].copy())
        # 如果提供了标签的预处理操作，则将其应用于标签数据。
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())
        # 将样本名称添加到样本字典中。
        sample['case_name'] = self.sample_list[idx].strip('\n')
        # 返回样本字典。
        return sample


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Synapse Training')
    parser.add_argument('--root_path', type=str, default='./data/Synapse/train_npz', help='root dir for train data')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
    args = parser.parse_known_args()[0]
    print(args)

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