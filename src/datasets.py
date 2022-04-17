import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from PIL import Image
import glob




class CustomDataset(Dataset):
    def __init__(self, patchpath_list, *args, **kwargs):
        super(CustomDataset, self)
        
        file_ids = [fp.split('/')[-1].split('.')[0] for fp in patchpath_list]
        file_names = ['-'.join(fi.split('-')[1:]) for fi in file_ids]
        file_classes = [fi.split('-')[0] for fi in file_ids]
        
        class_names = list(set(file_classes))
        class_names = sorted(class_names)
        class_labels = dict((name, i) for i, name in enumerate(class_names))
        
        self.class_labels = class_labels
        self.file_names = file_names
        self.X_path = patchpath_list
        self.y = [class_labels[cn] for cn in file_classes]

    def __getitem__(self, index):
        x = Image.open(self.X_path[index])
        y = self.y[index]
        filename = self.file_names[index]
        classname = [k for k,v in self.class_labels if v==y][0]
        
        x = torch.tensor(x).permute(2,0,1)
        y = torch.nn.functional.one_hot(torch.tensor(y), num_classes = len(self.class_labels))
        
        return {
            'x': x,
            'y': y,
            'filename': filename,
            'classname': classname
        }

    def __len__(self):
        return self.len

