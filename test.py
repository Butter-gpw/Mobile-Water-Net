import os
import torch
import time
import numpy as np
from datasets import TestDataset, denorm
from options import TestOptions
from util.util import AverageMeter, ProgressMeter
from torchvision import transforms
from models.mobile_water_net import *

class Predictor:
    def __init__(self, opt):
        self.opt = opt
        self.device = self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device(
            'cpu')
        self.model_path = self.opt.model
        self.save_path = self.opt.save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.print_freq = 20

        test_set = TestDataset(self.opt)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.opt.batch_size, shuffle=False,
                                                       num_workers=self.opt.workers)
        self.model = Mobile_Water_Net().to(self.device)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found!")
        self.load(self.model_path)

    def predict(self):
        self.model.eval()
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f'网络参数量：{num_params}')

        batch_time = AverageMeter("Time", "3.3f")
        progress = ProgressMeter(len(self.test_loader), [
            batch_time], prefix="Test: ")

        with torch.no_grad():
            end = time.time()
            for batch_idx, (paths, images) in enumerate(self.test_loader):
                bs = images.size(0)

                images = images.to(self.device)
                fake_images = self.model(images)

                fake_images = denorm(fake_images.data)
                fake_images = torch.clamp(fake_images, min=0., max=255.)
                fake_images = fake_images.type(torch.uint8)

                for idx in range(bs):
                    name = os.path.splitext(os.path.basename(paths[idx]))[0]
                    fake_image = fake_images[idx]
                    fake_image = transforms.ToPILImage()(fake_image).convert("RGB")
                    fake_image.save(f"{self.save_path}/{name}.png")

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % self.print_freq == 0:
                    progress.display(batch_idx)
        return

    def load(self, model):
        ckpt = torch.load(model, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        print(f"At epoch: {ckpt['epoch']} (loss={ckpt['best_loss']:.3f})")
        print(f">>> Load checkpoint from {model}")


if __name__ == "__main__":
    opt = TestOptions()
    opt = opt.initialize(opt.opt).parse_args()

    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)

    predictor = Predictor(opt)
    predictor.predict()
