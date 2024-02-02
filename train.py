import os
import torch
import time
import numpy as np
from datasets import PairDataset, split_data
from options import TrainOptions
from util import *
from models.ContrastLoss import *
from shutil import copyfile
from models.SSIMLoss import SSIM_LOSS
from datetime import datetime, timedelta, timezone
from models.mobile_water_net import *
import torch.optim as optim

"""
The code borrows heavily from the PyTorch implementation of FUnIE-GAN:
https://github.com/rowantseng/FUnIE-GAN-PyTorch
"""


class Trainer:
    def __init__(self, opt):
        self.opt = opt

        train, valid = split_data(self.opt.dataroot, valid_size=self.opt.valid_size, random_state=self.opt.random_seed)

        train_set = PairDataset(self.opt, train)
        valid_set = PairDataset(self.opt, valid)

        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.workers)
        self.valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=self.opt.batch_size, shuffle=False, num_workers=self.opt.workers)

        self.start_epoch = 0
        self.epochs = self.opt.epochs
        self.save_path = os.path.join(self.opt.checkpoints_dirs, self.opt.exp_name)
        os.makedirs(f"{self.save_path}", exist_ok=True)

        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.print_freq = 20
        self.best_loss = 1e6

        self.model = Mobile_Water_Net().to(self.device)

        if self.opt.model_resume:
            self.load(self.opt.model_resume)

        # loss function
        self.criterion = torch.nn.L1Loss().to(self.device)
        self.ssim_criterion = SSIM_LOSS().to(self.device)
        self.layer_infoNCE_loss = Contrast_Learning_Loss(self.device, self.opt.T, self.opt.requires_grad).to(self.device)

        self.gen_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), self.opt.lr)

    def train(self):
        for e in range(self.start_epoch, self.epochs):
            self.epoch = e
            _ = self.train_epoch()
            valid_gen_loss = self.validate()

            # Save models
            self.save(valid_gen_loss)

    def train_epoch(self):
        self.model.train()

        batch_time = AverageMeter("Time", "3.3f")
        Total_losses = AverageMeter("Total Loss")
        L1_losses = AverageMeter("L1 Loss")
        cl_losses = AverageMeter("Contrast Loss")
        ssim_losses = AverageMeter("SSIM Loss")
        progress = ProgressMeter(len(self.train_loader), [
            batch_time, Total_losses, L1_losses, cl_losses, ssim_losses], prefix="Train: ")

        end = time.time()
        for batch_idx, (ori_images, ref_images) in enumerate(self.train_loader):
            bs = ori_images.size(0)

            ori_images = ori_images.to(self.device)
            ref_images = ref_images.to(self.device)

            fake_images = self.model(ori_images)
            L1_loss = self.criterion(fake_images, ref_images)
            ssim_loss = self.ssim_criterion(fake_images, ref_images)

            # Contrast_Learning_Loss
            cl_loss = self.layer_infoNCE_loss(fake_images, ref_images, ori_images)

            # Total loss
            total_loss = (
                        self.opt.L1_loss_weight * L1_loss + self.opt.cl_loss_weight * cl_loss + self.opt.ssim_loss_weight * ssim_loss)
            self.gen_optimizer.zero_grad()
            total_loss.backward()
            self.gen_optimizer.step()

            # Update
            Total_losses.update(total_loss.item(), bs)
            L1_losses.update(L1_loss.item(), bs)
            cl_losses.update(cl_loss.item(), bs)
            ssim_losses.update(ssim_loss.item(), bs)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.print_freq == 0:
                progress.display(batch_idx)

        return Total_losses.avg

    def validate(self):
        self.model.eval()

        batch_time = AverageMeter("Time", "3.3f")
        Total_losses = AverageMeter("Total Loss")
        L1_losses = AverageMeter("L1 Loss")
        cl_losses = AverageMeter("Contrast Loss")
        ssim_losses = AverageMeter("SSIM Loss")

        progress = ProgressMeter(len(self.valid_loader), [
            batch_time, Total_losses, L1_losses, cl_losses, ssim_losses], prefix="Valid: ")

        with torch.no_grad():
            end = time.time()
            for batch_idx, (ori_images, ref_images) in enumerate(self.valid_loader):
                bs = ori_images.size(0)

                ori_images = ori_images.to(self.device)
                ref_images = ref_images.to(self.device)

                fake_images = self.model(ori_images)

                # Validate the generator
                L1_loss = self.criterion(fake_images, ref_images)
                cl_loss = self.layer_infoNCE_loss(fake_images, ref_images, ori_images)
                ssim_loss = self.ssim_criterion(fake_images, ref_images)

                # total loss
                Total_loss = (
                            self.opt.L1_loss_weight * L1_loss + self.opt.cl_loss_weight * cl_loss + self.opt.ssim_loss_weight * ssim_loss)

                Total_losses.update(Total_loss.item(), bs)
                L1_losses.update(L1_loss.item(), bs)
                cl_losses.update(cl_loss.item(), bs)
                ssim_losses.update(ssim_loss.item(), bs)

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % self.print_freq == 0:
                    progress.display(batch_idx)

        return Total_losses.avg

    def save(self, loss):
        # Check if the current model is the best
        is_best = loss < self.best_loss
        self.best_loss = min(self.best_loss, loss)

        # Prepare model info to be saved
        model_content = {"best_loss": loss, "epoch": self.epoch}
        model_path = f"{self.save_path}/{self.epoch}.pth.tar"

        # Save generator and discriminator
        model_content["state_dict"] = self.model.state_dict()
        torch.save(model_content, model_path)
        print(f">>> Save checkpiont to {model_path}")

        if is_best:
            copyfile(model_path, f"{self.save_path}/best.pth.tar")

    def load(self, model_resume):
        gen_ckpt = torch.load(model_resume, map_location=self.device)

        self.model.load_state_dict(gen_ckpt["state_dict"])
        self.best_loss = gen_ckpt["best_loss"]
        self.start_epoch = gen_ckpt["epoch"]

        print(f"At epoch: {self.start_epoch}")
        print(f">>> Load generator from {model_resume}")
        self.start_epoch += 1


if __name__ == '__main__':
    stat_time = datetime.now()


    opt = TrainOptions()
    opt = opt.initialize(opt.opt).parse_args()

    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)

    trainer = Trainer(opt)
    trainer.train()
    end_time = datetime.now()
    print(f"It takes {end_time - stat_time} to finish the training")
