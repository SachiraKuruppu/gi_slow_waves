'''
sw_at_datamodule.py

Datamodule to load slow waves to compute activation times.
'''

import os
import math

import torch as T
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from data_modules.SeqSlowWaveDataset_v3 import SeqSlowWaveDataset_v3

class SlowWaveATData_v3 (pl.LightningDataModule):
    def __init__ (self, data_dir:str, batch_size=16):
        '''Used to store info such as batch size, transformers.'''
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None) -> None:
        dataset_full = SeqSlowWaveDataset_v3(os.path.join(self.data_dir, 'annotations.csv'), self.data_dir, self.transforms)
        train_size = math.floor(0.8 * len(dataset_full))
        val_size = len(dataset_full) - train_size

        self.dataset_train, self.dataset_val = random_split(dataset_full, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=10)

    def predict_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=10)