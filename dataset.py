from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms, utils
import os
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


class SpyCamDataset(Dataset):
    """
    The Spy Cam dataset.
    """

    def __init__(self, csv_file, root_dir, use_orig=False, use_thermal=True, transform=None, orig_transform=None, binary=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            use_orig (bool, optional): Whether to include original image or not.
            transform (callable, optional): Optional transform to be applied on thermal images.
            orig_transform (callable, optional): Optional transform to be applied on original images.
        """
        self.csv = pd.read_csv(csv_file)
        self.csv = self.csv[self.csv['Photo Type'] == 'thermal']
        self.root_dir = root_dir
        self.use_orig = use_orig
        self.use_thermal = use_thermal
        self.transform = transform
        self.orig_transform = orig_transform
        self.binary = binary

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        """
        TODO: perform identical tranformation on both thermal and original images (e.g., random clip, flip)
        https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/48
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                str(self.csv.iloc[idx, 0]))
        try:
            image = io.imread(img_name)
        except FileNotFoundError:
            print("File with index={} not found.".format(idx))
        label = self.csv.iloc[idx, 3]

        if self.binary:
            label = label.astype(bool).astype(int)

        if self.transform:
            image = self.transform(image)

        if self.use_orig:

            ori_img_name = os.path.join(self.root_dir,
                                str(self.csv.iloc[idx, 1]).split('.')[0] + '-orig.png')
            try:
                # ori_image = io.imread(ori_img_name)
                ori_image = np.asarray(Image.open(ori_img_name).convert('RGB')) # need to convert RGBA image to RGB
            except FileNotFoundError:
                print("Original file with index={} not found. Missing file name: {}".format(idx, ori_img_name))
            if self.orig_transform:
                ori_image = self.orig_transform(ori_image)
            # image shape [1, 224, 224], ori_image shape [3, 224, 224]
            if not self.use_thermal:
                return ori_image, label

        if self.use_orig and self.use_thermal:
            image = torch.cat([image, ori_image], axis=0)

        return image, label


if __name__ == "__main__":
    dataset = SpyCamDataset(csv_file='/home/data/Spycam-AirBnb/AirBnb_metadata.csv',
                                    root_dir='/home/data/Spycam-AirBnb/RGB/',
                                    use_orig=True,
                                    binary=True,
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4695],
                                                              [0.1445])
                                    ]),
                                    orig_transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.3711, 0.3842, 0.3882],
                                                              [0.1395, 0.1488, 0.1646])
                                    ]))
    print(len(dataset))