import glob

import random

import numpy as np
import torch
from PIL import Image
import os
from sklearn.model_selection import train_test_split


def split_data(path, valid_size=0.1, random_state=42):
    dir = os.listdir(path)
    assert "trainA" in dir and "trainB" in dir
    trainA_path = os.path.join(path, "trainA")
    trainB_path = os.path.join(path, "trainB")
    trainA_list = os.listdir(trainA_path)
    trainB_list = os.listdir(trainB_path)
    assert trainA_list == trainB_list
    train, valid = train_test_split(trainA_list, test_size=valid_size, random_state=random_state)
    return train, valid

def norm(image):
    return (image / 127.5) - 1.0


def denorm(image):
    return (image + 1.0) * 127.5


def augment(dt_im, eh_im):
    # Random interpolation
    a = random.random()
    dt_im = dt_im * a + eh_im * (1 - a)

    # Random flip left right
    if random.random() < 0.25:
        dt_im = np.fliplr(dt_im)
        eh_im = np.fliplr(eh_im)

    # Random flip up down
    if random.random() < 0.25:
        dt_im = np.flipud(dt_im)
        eh_im = np.flipud(eh_im)

    return dt_im, eh_im


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, opt, split):
        super(PairDataset, self).__init__()
        self.opt = opt
        self.data_root = self.opt.dataroot
        self.im_size = self.opt.load_size
        self.split = split

        # Build image paths
        self.dt_ims = [f"{self.data_root}/trainA/{n}" for n in self.split]
        self.eh_ims = [f"{self.data_root}/trainB/{n}" for n in self.split]
        print(f"Total {len(self.dt_ims)} data")

    def __getitem__(self, index):
        # Read and resize image pair
        dt_im = Image.open(self.dt_ims[index]).convert("RGB")
        eh_im = Image.open(self.eh_ims[index]).convert("RGB")
        dt_im = dt_im.resize(self.im_size)
        eh_im = eh_im.resize(self.im_size)

        # Transfrom image pair to float32 np.ndarray
        dt_im = np.array(dt_im, dtype=np.float32)
        eh_im = np.array(eh_im, dtype=np.float32)

        # Augment image pair
        if self.split == "train":
            dt_im, eh_im = augment(dt_im, eh_im)

        # Transfrom image pair to (C, H, W) torch.Tensor
        dt_im = torch.Tensor(norm(dt_im)).permute(2, 0, 1)
        eh_im = torch.Tensor(norm(eh_im)).permute(2, 0, 1)
        return dt_im, eh_im

    def __len__(self):
        return len(self.dt_ims)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(TestDataset, self).__init__()
        self.opt = opt

        self.data_root = self.opt.dataroot
        self.im_size = self.opt.img_size
        self.ims = glob.glob(f"{self.data_root}/*")

    def __getitem__(self, index):
        # Read and resize image
        path = self.ims[index]
        im = Image.open(path).convert("RGB")
        im = im.resize(self.im_size)

        # Transfrom image to float32 np.ndarray
        im = np.array(im, dtype=np.float32)

        # Transfrom image to (C, H, W) torch.Tensor
        im = torch.Tensor(norm(im)).permute(2, 0, 1)
        return path, im

    def __len__(self):
        return len(self.ims)
