'''
sw_detection_datamodule.py

Data module to load slow wave data for detection. Can be used with both centered and shifted slow waves. The data should be separated out into 'train' and 'val' folders.

i.e. data_dir/train & data_dir/test
'''

import os

import torch as T
import pytorch_lightning as pl

from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

class SlowWaveDetectionData (pl.LightningDataModule):
    def __init__ (self, data_dir:str, batch_size=16):
        '''Used to store info such as batch size, transformers.'''
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None) -> None:
        data_generator = {k: datasets.ImageFolder(os.path.join(self.data_dir, k), self.transforms) for k in ['train', 'val']}

        self.data_loader = {k: DataLoader(data_generator[k], batch_size=self.batch_size, shuffle=(k == 'train'), num_workers=4) for k in ['train', 'val']}

    def train_dataloader(self):
        print('Number of training batches = %d' %(len(self.data_loader['train'])))
        return self.data_loader['train']

    def val_dataloader(self):
        print('Number of validation batches = %d' %(len(self.data_loader['val'])))
        return self.data_loader['val']

    def predict_dataloader(self):
        return self.data_loader['val']