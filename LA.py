import torch


class LA:
    def __init__(self):
        pass

    def __call__(self, img_fea, sk_fea):# 输入为（n，256，1024）(n, H*W, C)的特征图
        #print(len(img_fea.shape))
        img_fea = img_fea.permute(1, 0, 2)# （H*W，n，C）
        sk_fea = sk_fea.permute(1, 0, 2)# （H*W，n，C）
        local_dis = torch.cdist(sk_fea.contiguous(), img_fea.contiguous())
        #print(local_dis.shape)
        # [n,bs,bi]
        # 使用2norm计算距离
        dis = torch.norm(local_dis, p=2, dim=0)
        return dis

# test
if __name__ == '__main__':
    a = torch.randn(4, 256, 1024)
    b = torch.randn(4, 256, 1024)
    la = LA()
    dis = la(a, b)
    print(dis.shape)
    print(dis)