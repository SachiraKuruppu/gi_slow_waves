'''
SeqSlowWaveDataset.py

Pytorch dataset class to load the sequential slow wave data set.
'''

import os
import numpy as np
import pandas as pd
import torch as T
from torch.utils.data import Dataset
from PIL import Image

class SeqSlowWaveDataset (Dataset):
    def __init__ (self, annotations_file, img_dir, transform=None):
        
        self.image_labels = pd.read_csv(annotations_file, index_col=0)
        self.image_dir = img_dir
        self.transform = transform
    
    def __len__ (self):
        return len(self.image_labels)

    def __getitem__ (self, idx):
        img_name = self.image_labels.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path)
        label = self.image_labels.iloc[idx, 1:].tolist()
        is_slow_wave = label[0]
        at_time = label[2]

        if self.transform:
            image = self.transform(image)

        return image, is_slow_wave, at_time