import torch
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# get accuracy:acc_1
def get_acc_1(
        ph_branch,
        sk_branch,
        ph_loader,
        sk_loader,
        epoch,
        match,
):
    # set model to eval mode
    ph_branch.eval()
    sk_branch.eval()

    # start testing
    with torch.no_grad():
        # processing all photo
        tqdm_batch = tqdm(ph_loader, desc="Testing {}-epoch: photo-processing".format(epoch))
        
        # get photo and its label
        ph_list = []
        ph_label_list = []

        for ph, ph_label in tqdm_batch:
            ph = ph.cuda()
            
            # get output of model
            ph_fea = ph_branch(ph) # [n, 256, 1024]
            # print(ph_fea.shape)
            ph_fea = ph_fea.cpu() # 将张量转移到cpu上

            # get photo feature and its label
            for i in range(ph_fea.shape[0]):
                ph_list.append(ph_fea[i].unsqueeze(0))
                ph_label_list.append(ph_label[i])
        
        tqdm_batch.close()

        ph_feas = torch.cat(ph_list, dim=0)
        # print(ph_feas.shape)

        # retrival 
        sk_num = len(sk_loader.dataset)
        ph_num = len(ph_loader.dataset)
        # similarity matrix
        dist_mat = torch.zeros(sk_num, ph_num)

        correct_num = 0

        # processing all sketch
        tqdm_batch = tqdm(sk_loader, desc="Testing {}-epoch: sketch-processing".format(epoch))
        
        sk_label_list = []

        # sk_label's format is matchedId_repeatId,
        # e.g. 1_1 means the first sketch of the first class
        # e.g. 1_2 means the second sketch of the first class

        for i_sk, (sketch, label) in enumerate(tqdm_batch):
            sketch = sketch.cuda()
            
            # get output of model
            sk_fea = sk_branch(sketch) # [n, 256, 1024]

            # start retrival

            # i_sk is the index of sketch
            i_sk = i_sk * sk_loader.batch_size

            # j_sk is the index of sketch in a batch
            j_sk = i_sk + sk_fea.shape[0]

            # match sketch and photo by calculating distance
            for i_ph in range(0, ph_num, ph_loader.batch_size):
                
                # prevent out of index
                j_ph = min(i_ph + ph_loader.batch_size, ph_num)

                #print(ph_feas[i_ph:j_ph].shape)
                # calculate distance
                dist_mat[i_sk:j_sk, i_ph:j_ph] = match(
                    ph_feas[i_ph:j_ph].cuda(),
                    sk_fea.cuda(),
                ).cpu()

            # get sketch label
            for i in range(sketch.shape[0]):
                # only need the first number of label
                sk_label_list.append(label[i].split('_')[0])

        tqdm_batch.close()

        # get top1 accuracy
        # sort distance matrix
        indices = torch.sort(dist_mat)[1]# [sk_num, ph_num]
        # get top1 index

        # indices[:, 0] is the index of the most similar photo
        for i in range(sk_num):
            predict = ph_label_list[indices[i, 0]]

            # compare predict and label
            if predict == sk_label_list[i]:
                correct_num += 1

        acc_1 = correct_num / sk_num
        
    return acc_1




