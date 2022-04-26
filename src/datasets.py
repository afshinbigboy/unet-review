import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from PIL import Image
import glob



from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class ISIC2018Dataset(Dataset):
    """ISIC 2018 Dataset."""
    def __init__(self, root_dir, mode, transform=None, *args, **kwargs):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if mode.lower() not in ['train', 'validation', 'test']:
            raise ValueError("Invalid mode! Valid modes are: ['train', 'validation', 'test']")
        
        if mode.lower() == "train":
            pass
        if mode.lower() == "validation":
            pass
        if mode.lower() == "test":
            pass
        
        subject_names = ['-'.join(fn.split('-')[1:]) for fn in subjects]
        subjects_x = subjects
        subjects_y = subjects
        
        self.X_path_list = subjects_x
        self.Y_path_list = subjects_y
        
        self.output_size = output_size
        
    def __len__(self):
        return len(self.X_path_list)
    
    def __getitem__(self, index):
        x = Image.open(self.X_path_list[index])
        y = Image.open(self.Y_path_list[index])
        filename = self.file_names[index]
        
        x = torch.tensor(x).permute(2,0,1)
        if len(y.shape) < 3: 
            m = torch.tensor(y).unsqueeze(-1).permute(2,0,1)
        else: 
            m = torch.tensor(y).permute(2,0,1)
        
        # x and y must resize to the self.output_size
        sample = {'image': x, 'mask': m}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.file_names)