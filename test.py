import torch
from network import *
from dataset import *
from loss import *
from DLA import *
from LA import *
from eval import *
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ph_branch = LocalFeatureExtractor().to(device)
sk_branch = LocalFeatureExtractor().to(device)
match = DLA()

ckpt = torch.load('./results/checkpoint/DLA_LocalL2_high_frozeBN/ckpt-best.pth')
ph_branch.load_state_dict(ckpt['ph_encoder'])
sk_branch.load_state_dict(ckpt['sk_encoder'])

ph_test_root = './dataset/shoe_V2/test/photo'
sk_test_root = './dataset/shoe_V2/test/sketch'

ph_test_dataset = TestPhDataset(ph_test_root)
sk_test_dataset = TestSkDataset(sk_test_root)

ph_test_dataloader = DataLoader(
    ph_test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=12,
)

sk_test_dataloader = DataLoader(
    sk_test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=12,
)

# get acc_1
acc_1 = get_acc_1(
    ph_branch,
    sk_branch,
    ph_test_dataloader,
    sk_test_dataloader,
    99,
    match
)

print("acc_1: ", acc_1)
