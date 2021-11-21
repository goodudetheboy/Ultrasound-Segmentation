import torch
import torch.utils.data as data
import torchvision

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import random
import os

class UltrasoundSegmentation(data.Dataset):
    def __init__(self, folder_path):
        super(UltrasoundSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'bmode','*.png'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'mask',os.path.basename(img_path)))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data =  cv2.imread(img_path)
        label = cv2.imread(mask_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)