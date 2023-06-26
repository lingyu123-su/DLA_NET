# dataset位于./dataset/下,名为shoe_V2
# 该数据集包含了两个文件夹,分别为train和test
# 存放关系如下：
# -train
#   -photo
#     1.png
#     2.png
#     ...
#   -sketch
#     1.png
#     2.png
#     ...
# -test
#   -photo
#     1.png
#     2.png
#     ...
#   -sketch
#     1.png
#     2.png
#     ...
# 草图和图片的对应关系见sketch_photo_id.mat
#  sketch_photo_id.mat中字典‘photo_id_train’和‘photo_id_test’分别存储训练集和测试集草图和图片的对应关系，两者格式一样。
# 以字典‘photo_id_test’为例说明，data[‘photo_id_test’]数据形式为666*1的数组，data[‘photo_id_test’][0]为test/sketch/1.png对应的photo文件名，data[‘photo_id_test’][1]为test/sketch/2.png对应的photo文件名。
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import scipy.io as sio
from glob import glob
import random
from torch.utils.tensorboard import SummaryWriter



# 定义数据集
class TrainDataset(data.Dataset):
    def __init__(self, photo_root, sketch_root):
        self.photo_root = photo_root
        self.sketch_root = sketch_root
        # 读入数据集的图片名
        self.photo_list = os.listdir(photo_root)
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
        # 对图片进行预处理
        self.transform = transforms.Compose([
            transforms.Resize(288),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            norm
        ])

    def __getitem__(self, index):
        photo_name = self.photo_list[index]
        photo_label = photo_name.split('.')[0]
        photo_path = os.path.join(self.photo_root, photo_name)
        photo = Image.open(photo_path)
        # 如果图片是单通道的，转换成三通道
        if len(photo.split()) == 1:
            photo = photo.Merge("RGB", (photo, photo, photo))
        photo = self.transform(photo)

        # 获取sketch和photo的对应关系
        sketch_patern = os.path.join(self.sketch_root, photo_label + '_*')
        sketch_paths = glob(sketch_patern)
        random.shuffle(sketch_paths)
        sketch_path = sketch_paths[0]# 随机选择一个sketch，和photo组成一组数据
        sketch = Image.open(sketch_path)
        # 如果sketch只有一通道，转换成三通道
        if len(sketch.split()) == 1:
            sketch = sketch.Merge("RGB", (sketch, sketch, sketch))
        sketch = self.transform(sketch)
        
        return photo, sketch
    
    def __len__(self):
        return len(self.photo_list)
    
    # 测试图片集
class TestPhDataset(data.Dataset):
    def __init__(self, photo_root):
        self.photo_root = photo_root
        self.photo_list = os.listdir(photo_root)
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            norm
        ])

    def __getitem__(self, index):
        photo_name = self.photo_list[index]
        photo_label = photo_name.split('.')[0]
        photo_path = os.path.join(self.photo_root, photo_name)
        photo = Image.open(photo_path)
        if len(photo.split()) == 1:
            photo = photo.Merge("RGB", (photo, photo, photo))
        photo = self.transform(photo)
        return photo, photo_label
    
    def __len__(self):
        return len(self.photo_list)
    

# 测试草图集
class TestSkDataset(data.Dataset):
    def __init__(self, sketch_root):
        self.sketch_root = sketch_root
        self.sketch_list = os.listdir(sketch_root)
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            norm
        ])

    def __getitem__(self, index):
        sketch_name = self.sketch_list[index]
        sketch_label = sketch_name.split('.')[0]
        sketch_path = os.path.join(self.sketch_root, sketch_name)
        sketch = Image.open(sketch_path)
        if len(sketch.split()) == 1:
            sketch = sketch.Merge("RGB", (sketch, sketch, sketch))
        sketch = self.transform(sketch)
        return sketch, sketch_label
    
    def __len__(self):
        return len(self.sketch_list)

if __name__ == '__main__':
    # 测试数据集
    photo_root = './dataset/shoe_V2/train/photo'
    sketch_root = './dataset/shoe_V2/train/sketch'
    dataset = TrainDataset(photo_root, sketch_root)
    print(len(dataset))
   # 使用tensorboard可视化前10张图片
    for i in range(10):
        photo, sketch = dataset[i]
        # 使用tensorboard可视化
        writer = SummaryWriter('./logs/tensorboard/train_dataset')
        writer.add_image('photo', photo, i)
        writer.close()
        writer = SummaryWriter('./logs/tensorboard/train_dataset')
        writer.add_image('sketch', sketch, i)
        writer.close()
    print('------------------')
    # 测试数据集
    photo_root = './dataset/shoe_V2/test/photo'
    dataset = TestPhDataset(photo_root)
    print(len(dataset))
   # 使用tensorboard可视化前10张图片
    for i in range(10):
        photo, photo_label = dataset[i]
        # 使用tensorboard可视化
        writer = SummaryWriter('./logs/tensorboard/test_dataset')
        writer.add_image('photo', photo, i)
        writer.close()
    print('------------------')
    # 测试数据集
    sketch_root = './dataset/shoe_V2/test/sketch'
    dataset = TestSkDataset(sketch_root)
    print(len(dataset))
    # 使用tensorboard可视化前10张图片
    for i in range(10):
        sketch, sketch_label = dataset[i]
        # 使用tensorboard可视化
        writer = SummaryWriter('./logs/tensorboard/test_dataset')
        writer.add_image('sketch', sketch, i)
        writer.close()