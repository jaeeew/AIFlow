from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from torchvision import transforms
import torch
from PIL import Image

class ContamDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 이미지 파일 리스트
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지와 마스크 로드
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert("RGB")  # 이미지 로드 (RGB)
        mask = Image.open(mask_path).convert("L")  # 마스크는 흑백으로 로드 (L)

        # transform이 주어졌다면 적용
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
