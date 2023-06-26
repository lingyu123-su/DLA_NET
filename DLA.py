import torch


class DLA:
    def __init__(self):
        pass

    def __call__(self, ph_fea, sk_fea):
        bi, n, _ = ph_fea.shape 
        # bi is batch size, n is H*W
        bs = sk_fea.shape[0]
        # bs is batch size

        # ph_fea [bi,n,f]（batch_size，H*W，C）
        ph_fea = ph_fea.permute(2, 0, 1).flatten(1)
        # [f,bi*n],将ph_fea的维度顺序改为（C，batch_size*H*W）,然后展平
        ph_fea = ph_fea.permute(1, 0).unsqueeze(0)
        # [1,bi*n,f],在ph_fea的第0维增加一个维度,变为（1，batch_size*H*W，C）
        sk_fea = sk_fea.permute(2, 0, 1).flatten(1)
        # [f,bs*n],将sk_fea的维度顺序改为（C，batch_size*H*W）,然后展平
        sk_fea = sk_fea.permute(1, 0).unsqueeze(0)
        # [1,bs*n,f] 在sk_fea的第0维增加一个维度,变为（1，batch_size*H*W，C）
        local_dis = torch.cdist(sk_fea, ph_fea)[0]
        # [bs*n,bi*n],计算sk_fea和ph_fea的距离
        local_dis = local_dis.reshape([bs, n, bi, n])
        # [bs,n,bi,n].将local_dis的维度变为（batch_size，H*W，batch_size，H*W）
        local_dis = torch.min(local_dis, dim=3)[0]
        # [bs,n,bi].对local_dis的第3维进行最小值池化
        local_dis = local_dis.permute(1, 0, 2)
        # [n,bs,bi],将local_dis的维度顺序改为（H*W，batch_size，batch_size）
        dis = torch.norm(local_dis, p=2, dim=0)# 使用2norm计算距离
        return dis

if __name__=='__main__':
    match = DLA()
    ph_fea = torch.rand(4,256,1024)
    sk_fea = torch.rand(4,256,1024)
    dis = match(ph_fea,sk_fea)
    print(dis.shape)
    print(dis)