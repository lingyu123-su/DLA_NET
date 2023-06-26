# train the model
import torch
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestPhDataset, TestSkDataset
from network import LocalFeatureExtractor
from loss import triplet_loss
from DLA import DLA
from LA import LA
from tqdm import tqdm
from utils.ckpt_utils import *
from utils.Logger import *
from eval import *
from options.train_options import TrainOptions
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # options
    opt = TrainOptions().parse()
    logger = Logger(opt.logger_path).get_logger()
    logger.info(opt)

    # model
    local_photo_feature_extractor = LocalFeatureExtractor().to(device)
    local_sketch_feature_extractor = LocalFeatureExtractor().to(device)

    # hyper-parameters
    lr = opt.lr
    epoch = 0
    batch_size = opt.batch_size
    best_acc = 0
    logger.info("successfully build model!")
    # DLA or LA
    if opt.match == "DLA":
        match = DLA()
    else:
        match = LA()

    # dataset
    train_dataset = TrainDataset('./dataset/shoe_V2/train/photo', './dataset/shoe_V2/train/sketch')
    test_ph_dataset = TestPhDataset('./dataset/shoe_V2/test/photo')
    test_sk_dataset = TestSkDataset('./dataset/shoe_V2/test/sketch')
    # dataloader
    logger.info("start loading training data...")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=12,
        drop_last=True,
    )
    logger.info("start loading testing data...")
    test_ph_dataloader = DataLoader(
        test_ph_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
    )
    test_sk_dataloader = DataLoader(
        test_sk_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
    )

    # optimizer
    parameters = itertools.chain(
        local_photo_feature_extractor.parameters(),
        local_sketch_feature_extractor.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))

    # loss
    loss_tracker = AverageMeter()

    # start training
    logger.info("start training...")
    # train loop
    for i in range(epoch + 1, opt.epoch + 1):

        local_photo_feature_extractor.train()
        local_sketch_feature_extractor.train()
        
        # reminder
        tqdm_batch = tqdm(train_dataloader, desc="Training epoch: " + str(i))

        for photo, sketch in tqdm_batch:
            photo = photo.to(device)
            sketch = sketch.to(device)

            # get local features
            local_photo_feature = local_photo_feature_extractor(photo)
            local_sketch_feature = local_sketch_feature_extractor(sketch)

            # calculate triplet loss
            loss = triplet_loss(match(
                local_photo_feature,
                local_sketch_feature,
            ), opt.margin)
            loss_tracker.update(loss.item(), photo.shape[0])

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(
            "Epoch: {}, Average loss: {:.4f}, loss: {:8f}".format(i, loss_tracker.avg, loss_tracker.val)
        )
        tqdm_batch.close()

        # test
        acc_1 = get_acc_1(
            local_photo_feature_extractor,
            local_sketch_feature_extractor,
            test_ph_dataloader,
            test_sk_dataloader,
            i,
            match,
        )

        # save model
        if acc_1 > best_acc:
            best_acc = acc_1
            logger.info("best acc: {:.4f}".format(best_acc))
            models = [
                local_photo_feature_extractor,
                local_sketch_feature_extractor
            ]
            names = ["ph_encoder", "sk_encoder"]
            save_model(opt, models, names, i, logger, best=True)

        

if __name__ == "__main__":
    train()
