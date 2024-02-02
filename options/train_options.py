import argparse


class TrainOptions:
    def __init__(self):
        self.opt = argparse.ArgumentParser()

    def initialize(self, parser):
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use '' for CPU')
        parser.add_argument("--epochs", default=300, type=int, metavar="N",
                            help="number of total epochs to run")
        parser.add_argument("--random_seed", default=42, type=int, metavar="N",
                            help="random_seed for the project")
        parser.add_argument("--valid_size", default=0.1, type=float or int, metavar="N",
                            help="float or int, default=0.1 If float, should be between 0.0 and 1.0 "
                                 "and represent the proportion of the dataset to include in the valid split. "
                                 "If int, represents the absolute number of valid samples."
                            )
        # dataset parameters
        parser.add_argument('--load_size', type=tuple, default=(256, 256), help='scale images to this size')
        parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                            help="number of data loading workers (default: 8)")


        # path
        parser.add_argument("--model_resume", default="", type=str, metavar="PATH",
                            help="path to latest model checkpoint (default: none)")
        parser.add_argument('--dataroot', default='', type=str,
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--checkpoints_dirs', type=str, default='./checkpoints/',
                            help='experiment dirs are saved here')
        parser.add_argument('--exp_name', type=str, default='',
                            help='experiment dirs where checkpoints are saved')

        parser.add_argument("-b", "--batch_size", default=24, type=int,
                            metavar="N",
                            help="mini-batch size, this is the total "
                                 "batch size of all GPUs on the current node when "
                                 "using Data Parallel or Distributed Data Parallel")
        parser.add_argument("--lr", "--learning-rate", default=0.0001, type=float,
                            metavar="LR", help="initial learning rate")

        # loss functions
        parser.add_argument("--requires_grad", default=False, type=bool,
                            help="Train contrast_Learning_feature extractor")
        parser.add_argument("--L1_loss_weight", default=1, type=float, help="L1_loss weight in training")
        parser.add_argument("--ssim_loss_weight", default=1, type=float, help="ssim_loss_weight in training")
        parser.add_argument("--cl_loss_weight", default=0.2, type=float, help="cl_loss_weight in training")
        parser.add_argument("--T", default=0.07, type=float, help="softmax temperature")

        return parser
