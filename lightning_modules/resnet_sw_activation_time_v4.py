'''
resnet_sw_activation_time.py

Resetnet regression/classifier model to predict the slow wave activation times from scalograms.
'''

import torch as T
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
import torchmetrics.functional as tF

from pytorch_lightning.loggers import TensorBoardLogger

class ResnetSlowWaveAT_v4 (pl.LightningModule):

    def __init__ (self):
        super().__init__()

        backbone = models.resnet50(pretrained=True)
        num_features = backbone.fc.in_features

        layers = list(backbone.children())[:-1]
        self.feature_extractor = T.nn.Sequential(*layers)

        self.classifier = T.nn.Linear(num_features, 2)
        self.regressor = T.nn.Linear(num_features, 1)

    def forward (self, x):
        features = self.feature_extractor(x).flatten(1)
        sw_class = self.classifier(features)
        sw_AT    = self.regressor(features)
        return sw_class, sw_AT

    def __calculate_loss (self, y_class, y_AT, y_hat_class, y_hat_AT):
        y_hat_AT = T.reshape(y_hat_AT, (-1, 1))
        y_AT = T.reshape(y_AT, (-1, 1))

        # Remove the loss from scalogram windows without slow waves
        y_hat_AT[y_class == -1] = -1

        loss = F.cross_entropy(y_hat_class, y_class)
        reg_loss = F.mse_loss(y_hat_AT, y_AT)
        loss += reg_loss
        
        acc = tF.accuracy(y_hat_class, y_class)

        return loss, reg_loss, acc


    def training_step (self, batch, batch_idx):
        _, x, y_class, _, y_AT = batch
        y_AT = y_AT.float()
        
        y_hat_class, y_hat_AT = self(x) # calling forward()
        
        loss, reg_loss, acc = self.__calculate_loss(y_class, y_AT, y_hat_class, y_hat_AT)

        self.logger.experiment.add_scalar('Loss/Train', loss, batch_idx)
        self.logger.experiment.add_scalar('Accuracy/Train', acc, batch_idx)
        self.logger.experiment.add_scalar('Regression Loss/Train', reg_loss, batch_idx)
        
        return loss

    def validation_step (self, batch, batch_idx):
        _, x, y_class, _, y_AT = batch
        y_AT = y_AT.float()

        y_hat_class, y_hat_AT = self(x) # calling forward()

        loss, reg_loss, acc = self.__calculate_loss(y_class, y_AT, y_hat_class, y_hat_AT)

        self.logger.experiment.add_scalar('Loss/Val', loss, batch_idx)
        self.logger.experiment.add_scalar('Accuracy/Val', acc, batch_idx)
        self.logger.experiment.add_scalar('Regression Loss/Val', reg_loss, batch_idx)

        return loss

    def predict_step(self, batch, batch_idx):
        x_name, x, y_class, y_num_sw, y_AT = batch
        y_hat_class, y_hat_AT = self(x)
        class_probs = F.softmax(y_hat_class, dim=1)

        return x_name, y_class, y_num_sw, y_AT, class_probs, y_hat_AT

    def configure_optimizers (self):
        optimizer = T.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer