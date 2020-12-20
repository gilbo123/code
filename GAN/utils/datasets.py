import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class ImageDataset(Dataset):
    def __init__(self, orig, proc, shape):
        self.orig_files = sorted(glob.glob(orig + "*.*"))
        self.proc_files = sorted(glob.glob(proc + "*.*"))
        self.transform = transforms.Compose(
            [
                #transforms.Resize(shape, Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        assert len(self.orig_files) == len(self.proc_files), "Unequal number of files!"

    def __getitem__(self, index):
        orig_img = Image.open(self.orig_files[index % len(self.orig_files)])
        proc_img = Image.open(self.proc_files[index % len(self.proc_files)])

        return {"orig": self.transform(orig_img), "proc": self.transform(proc_img)}

    def __len__(self):
        return len(self.orig_files)# + len(self.proc_files)
