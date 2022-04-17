from extra.utils import (
  load_config,
  _print,
)
import json
from datasets import CustomDataset


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
# from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import numpy as np
import copy


import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from PIL import Image
import glob


config = load_config("./configs/default.yaml")
_print("Config:", "info_bold")
print(json.dumps(config, indent=2))
print(20*"~-", "\n")


custom_img = CustomDataset("")
# total images in set
print(custom_img.len)

train_len = int(0.6*custom_img.len)
test_len = custom_img.len - train_len
train_set, test_set = CustomDataset.random_split(custom_img, lengths=[train_len, test_len])
# check lens of subset
len(train_set), len(test_set)

train_set = CustomDataset("")
train_set = torch.utils.data.TensorDataset(train_set, train=True, batch_size=4)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=1)
print(train_set)
print(train_loader)

test_set = torch.utils.data.DataLoader(Dataset, batch_size=4, sampler=test_set)
test_loader = torch.utils.data.DataLoader(Dataset, batch_size=4)

print(config)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
# optimizer = optim.RMSprop(Net.parameters(), lr= float(config['lr']), weight_decay=1e-8, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
model.to(device)
if int(pretrained=True):
    model.load_state_dict(torch.load(saved_model_path, map_location='cpu')['model_weights'])