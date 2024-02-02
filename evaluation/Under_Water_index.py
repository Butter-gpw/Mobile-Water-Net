import cv2
import pandas as pd
import numpy as np
import os
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from uqim_utils import getUIQM
from UCIQE import uciqe

def ssim(im1, im2):
    return structural_similarity(im1, im2, channel_axis=2)


def UIQM(img):
    return getUIQM(np.array(img))


class Under_Water_index_Proceesing:
    def __init__(self, tested_path, ref_path=None, ref_index=True, non_ref_index=True, img_size=(512, 512),
                 reference_list=None, non_reference_list=None):
        self.tested_path = tested_path  # 被测图片文件夹路径
        self.ref_path = ref_path  # 参考图片文件夹路径
        self.ref_index = ref_index  # 是否使用参照指标
        self.non_ref_index = non_ref_index  # 是否使用非参照指标
        self.img_size = img_size  # 进行图片缩放
        self.MSE = mean_squared_error
        self.PSNR = peak_signal_noise_ratio
        self.SSIM = ssim
        self.UIQM = UIQM
        self.uciqe = uciqe
        self.reference_list = [self.MSE, self.PSNR, self.SSIM] if reference_list is None else reference_list
        self.non_reference_list = [self.UIQM, self.uciqe] if non_reference_list is None else non_reference_list
        self.form = None

    def calculate(self):
        tested_img_list = os.listdir(self.tested_path)
        tested_img_list.sort()

        # print(tested_img_list)
        if self.ref_index:
            ref_img_list = os.listdir(self.ref_path)
            ref_img_list.sort()
            assert tested_img_list == ref_img_list, "The tested image and reference image should have the same name and file format (e.g.. Jpg. png)"
        for i, img_name in tqdm(enumerate(tested_img_list), total=len(tested_img_list)):
            tested_img_path = os.path.join(self.tested_path, img_name)
            index_dict = {"name": img_name.split('.')[0]}
            img_tested = cv2.imread(tested_img_path)
            img_tested = cv2.cvtColor(img_tested, cv2.COLOR_BGR2RGB)
            if self.img_size:
                img_tested = cv2.resize(img_tested, self.img_size)
            # img_tested = Image.open(tested_img_path)
            # img_ref = Image.open(ref_img_path)
            if self.ref_index:
                ref_img_path = os.path.join(self.ref_path, img_name)
                img_ref = cv2.imread(ref_img_path)
                img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
                if self.img_size:
                    img_ref = cv2.resize(img_ref, self.img_size)
                for ref_fun in self.reference_list:
                    index_dict[ref_fun.__name__] = ref_fun(img_tested, img_ref)
                del img_ref
            if self.non_ref_index:
                for non_ref_fun in self.non_reference_list:
                    index_dict[non_ref_fun.__name__] = non_ref_fun(img_tested)
            del img_tested
            df_temp = pd.DataFrame(index_dict, index=[i])
            self.form = df_temp if self.form is None else pd.concat([self.form, df_temp])
