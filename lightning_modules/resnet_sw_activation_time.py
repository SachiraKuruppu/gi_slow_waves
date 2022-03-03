'''
resnet_sw_activation_time.py

Resetnet regression/classifier model to predict the slow wave activation times from scalograms.
'''

import torch as T
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
import torchmetrics.functional as tF

class ResnetSlowWaveAT (pl.LightningModule):

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

        loss = F.cross_entropy(y_hat_class, y_class)
        reg_loss = F.mse_loss(y_hat_AT, y_AT)
        loss += reg_loss
        
        acc = tF.accuracy(y_hat_class, y_class)

        return loss, reg_loss, acc


    def training_step (self, batch, batch_idx):
        x, y_class, y_AT = batch
        y_AT = y_AT.float()
        
        y_hat_class, y_hat_AT = self(x) # calling forward()
        
        loss, reg_loss, acc = self.__calculate_loss(y_class, y_AT, y_hat_class, y_hat_AT)

        self.log('train_loss', loss)
        self.log('train_class_acc', acc)
        if reg_loss is not None:
            self.log('train_reg_loss', reg_loss)
        
        return loss

    def validation_step (self, batch, batch_idx):
        x, y_class, y_AT = batch
        y_AT = y_AT.float()

        y_hat_class, y_hat_AT = self(x) # calling forward()

        loss, reg_loss, acc = self.__calculate_loss(y_class, y_AT, y_hat_class, y_hat_AT)

        self.log('val_loss', loss)
        self.log('val_class_acc', acc)
        if reg_loss is not None:
            self.log('val_reg_loss', reg_loss)
        
        return loss

    def predict_step(self, batch, batch_idx):
        x, y_class, y_AT = batch
        y_hat_class, y_hat_AT = self(x)
        class_probs = F.softmax(y_hat_class, dim=1)

        return y_class, y_AT, class_probs, y_hat_AT

    def configure_optimizers (self):
        optimizer = T.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer