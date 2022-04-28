from __future__ import print_function, division

import os
import pandas as pd
import numpy as np
import glob
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib.image import imread


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode



class ISIC2018Dataset(Dataset):
	"""ISIC 2018 Dataset."""
	def __init__(
		self, 
		root_dir, 
		x_folder,
		x_filename_format,
		y_folder=None,
		y_filename_format=None,
		transform_list=[], 
		*args, **kwargs):
		"""Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir if not '/' == root_dir[-1] else os.path.dirname(root_dir)
		self.x_fp_list = glob.glob(f"{os.path.join(self.root_dir, x_folder)}/{x_filename_format}")
		self.fid_list = [fn.split('/')[-1].split('.')[0].split('_')[1] for fn in self.x_fp_list]
		
		self.x_folder = x_folder
		self.x_filename_format = x_filename_format
		self.y_folder = y_folder
		self.y_filename_format = y_filename_format

		if transform_list:
			self.transform = transforms.Compose([eval(t) for t in transform_list])
		else:
			self.transform = None
		

	def __len__(self):
		return len(self.fid_list)


	def __getitem__(self, index):
		file_id = self.fid_list[index]

		x_filepath = f"{os.path.join(self.root_dir, self.x_folder)}/{self.x_filename_format.replace('*', file_id)}"
		x = imread(x_filepath)
		x = torch.tensor(x).permute(2,0,1)
		
		if self.y_folder:
			y_filepath = f"{os.path.join(self.root_dir, self.y_folder)}/{self.y_filename_format.replace('*', file_id)}"
			y = imread(y_filepath)
			if len(y.shape) < 3: 
				y = torch.tensor(y).unsqueeze(-1).permute(2,0,1)
			else: 
				y = torch.tensor(y).permute(2,0,1)
			sample = {'x': x, 'y': y}
		else:
			sample = {'x': x}

		if self.transform:
			sample = self.transform(sample)

		return sample
