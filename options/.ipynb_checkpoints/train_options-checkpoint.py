import argparse
import os
import datetime


class TrainOptions:
    def __init__(self):
        parser = argparse.ArgumentParser()

        # dataset parameters
        parser.add_argument("--ph_train_root", help="the root directory of train photo")
        parser.add_argument(
            "--sk_train_root", help="the root directory of train sketch"
        )
        parser.add_argument("--ph_test_root", help="the root directory of test photo")
        parser.add_argument("--sk_test_root", help="the root directory of test sketch")
        # model parameters
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument(
            "--margin", type=float, default=0.3, help="margin value of triplet loss"
        )
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument(
            "--epoch", type=int, default=100, help="max num of train epoch"
        )
        parser.add_argument(
            "--norm_type",
            type=str,
            help="norm type used to sum local distances, 2norm || 1norm",
        )
        # save path parameters
        parser.add_argument("--result_dir", required=True, help="directory of result")
        self.opt = parser.parse_args()

    def make_save_dir(self):
        now_str = datetime.datetime.now().__str__().replace(" ", "_")
        result_path = self.opt.result_dir
        log_dir = os.path.join(result_path, "log")
        checkpoint_dir = os.path.join(result_path, "checkpoint")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.opt.logger_path = os.path.join(log_dir, now_str + ".log")


        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.opt.checkpoint_path = os.path.join(checkpoint_dir, now_str)

    def parse(self):
        self.make_save_dir()
        return self.opt
