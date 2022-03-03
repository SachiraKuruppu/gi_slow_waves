'''
resnet_sw_detect.py

Restnet model to detect slow waves from scalograms. This model can be trained on both centered slow waves and shifted slow waves.
'''

import torch as T
import pytorch_lightning as pl
import torchvision.models as models
import torchmetrics

class ResnetSlowWave (pl.LightningModule):

    def __init__ (self):
        super().__init__()

        backbone = models.resnet50(pretrained=True)
        num_features = backbone.fc.in_features

        layers = list(backbone.children())[:-1]
        self.feature_extractor = T.nn.Sequential(*layers)

        self.classifier = T.nn.Linear(num_features, 2)

        self.criterion = T.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.softmax = T.nn.Softmax(dim=1)

    def forward (self, x):
        features = self.feature_extractor(x).flatten(1)
        return self.classifier(features)

    def training_step (self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # calling forward()
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y))
        return loss

    def validation_step (self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # calling forward()
        val_loss = self.criterion(y_hat, y)

        self.log('val_loss', val_loss)
        self.log('val_acc', self.accuracy(y_hat, y))
        return val_loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        probs = self.softmax(y_hat)
        return y, probs

    def configure_optimizers (self):
        optimizer = T.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    