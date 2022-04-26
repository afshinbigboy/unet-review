import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from PIL import Image
import glob




class ISIC2018Dataset(Dataset):
    def __init__(self, subjects, output_size, *args, **kwargs):
        super(ISIC2018Dataset, self)
        
        subject_names       = ['-'.join(fn.split('-')[1:]) for fn in subjects]
        subjects_x          = subjects
        subjects_y          = subjects
        
        self.X_path_list    = subjects_x
        self.Y_path_list    = subjects_y
        
        self.output_size    = output_size

    def __getitem__(self, index):
        x = Image.open(self.X_path_list[index])
        y = Image.open(self.Y_path_list[index])
        filename = self.file_names[index]
        
        x = torch.tensor(x).permute(2,0,1)
        if len(y.shape) < 3: 
            y = torch.tensor(y).unsqueeze(-1).permute(2,0,1)
        else: 
            y = torch.tensor(y).permute(2,0,1)
        
        # x and y must resize to the self.output_size
        return {
            'x': x,
            'y': y,
            'filename': filename,
        }

    def __len__(self):
        return len(self.file_names)