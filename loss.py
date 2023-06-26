import torch


# 输入是（n,n）的距离矩阵,训练时，对角线上的元素为正样本，其余为负样本
# 例如
# 1 -1 -1 -1
# -1 1 -1 -1
# -1 -1 1 -1
# -1 -1 -1 1
def triplet_loss(dis, margin):
        b=dis.shape[0]
        # [b,b]

        eye = torch.eye(b).cuda()
        pos_dis = torch.masked_select(dis, eye.bool())
        # [b]
        loss = pos_dis.unsqueeze(1) - dis + margin
        loss = loss * (1 - eye)
        loss = torch.nn.functional.relu(loss)

        hard_triplets = loss > 0
        # [b,b]
        num_pos_triplets = torch.sum(hard_triplets, dim=1)
        # [b]
        loss = torch.sum(loss, dim=1) / (num_pos_triplets + 1e-16)
        loss = torch.mean(loss)
        return loss

# test
if __name__ == '__main__':
        a = torch.randn(4, 4)
        print(a)
        loss = triplet_loss(a, 0.1)
        print(loss)