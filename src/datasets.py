import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from PIL import Image
import glob




class ISIC2020Dataset(Dataset):
    def __init__(self, root_directory, *args, **kwargs):
        super(ISIC2020Dataset, self)
        folder_names    = glob.glob(f"{root_directory}")
        file_names      = ['-'.join(fn.split('-')[1:]) for fn in folder_names]
        
        self.file_names = file_names
        self.X_path     = root_directory
        self.Y_path     = root_directory

    def __getitem__(self, index):
        x = Image.open(self.X_path[index])
        y = Image.open(self.Y_path[index])
        filename = self.file_names[index]
        
        x = torch.tensor(x).permute(2,0,1)
        if len(y.shape) < 3: 
            y = torch.tensor(y).unsqueeze(-1).permute(2,0,1)
        else: 
            y = torch.tensor(y).permute(2,0,1)
        
        return {
            'x': x,
            'y': y,
            'filename': filename,
        }

    def __len__(self):
        return len(self.file_names)
