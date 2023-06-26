# 此文件对DLA_NET的网络结构进行了定义
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from torch.autograd import Variable

# 定义DLA网络的基本模块
class LocalFeatureExtractor(nn.Module):
    def __init__(self):
        super(LocalFeatureExtractor, self).__init__()
        resnet50 = models.resnet50(pretrained=True)

        # middle feature map
        self.backbone = nn.Sequential(
            OrderedDict(list(resnet50.named_children())[:-3]))
        self.wh = 16 # the shape of output feature map is 16*16
        self.channels = 1024 # the number of channels of output feature map is 1024

        # # high feature map
        # self.backbone = nn.Sequential(
        #     OrderedDict(list(resnet50.named_children())[:-2]))
        # self.wh = 8 # the shape of output feature map is 8*8
        # self.channels = 2048 # the number of channels of output feature map is 2048

        # # 固定住bn层的参数
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                # m.weight.requires_grad = False
                # m.bias.requires_grad = False
                
        

    def forward(self, x):
        feature = self.backbone(x)
        #print(feature.shape) #conv3的输出为torch.Size([1, 1024, 16, 16])
        # # local feature L2 normalization，对每个16*16的特征图进行L2归一化
        feature = torch.nn.functional.normalize(feature, dim=1)

        # global feature L2 normalization，对整个1024*16*16的特征图进行L2归一化
        # feature = torch.nn.functional.normalize(feature, dim=0)

        #print(feature.shape) #torch.Size([1, 1024, 16, 16])
        out = feature.flatten(2).permute(0, 2, 1)# 对特征图 fea 先进行展平操作，然后改变其维度顺序，得到新的张量 out。
        return out # out的维度为[n, 256, 1024](n, H*W, C)

# test
if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256)
    model = LocalFeatureExtractor()
    out = model(x)
    print(out.shape)

