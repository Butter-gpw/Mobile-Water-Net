import argparse

class TestOptions:
    def __init__(self):
        self.opt = argparse.ArgumentParser()

    def initialize(self, parser):
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use '' for CPU')
        parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                            help="number of data loading workers (default: 8)")
        parser.add_argument("--random_seed", default=42, type=int, metavar="N",
                            help="random_seed for the project")

        # model parameters

        parser.add_argument('--dataroot', default='', type=str,
                            help='path to tested images')
        parser.add_argument("-b", "--batch_size", default=16, type=int,
                            metavar="N",
                            help="mini-batch size (default: 16), this is the total "
                                 "batch size of all GPUs on the current node when "
                                 "using Data Parallel or Distributed Data Parallel")

        parser.add_argument("-m", "--model",
                            default="",
                            type=str, metavar="PATH",
                            help="path to generator checkpoint (default: none)")

        parser.add_argument("--save_path",
                            default="", type=str,
                            metavar="PATH",
                            help="path to save results (default: none)")

        parser.add_argument("--img_size", default=(256, 256), type=tuple, help="scale images to this size")

        return parser
