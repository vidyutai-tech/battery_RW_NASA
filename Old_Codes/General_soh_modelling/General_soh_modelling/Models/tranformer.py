import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from data_loading import data_module
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class TransformerModel(pl.LightningModule):
    def __init__(self, input_size, num_layers,num_heads,lr):
        super(TransformerModel, self).__init__()

        #self.embedding= nn.Embedding(input_size, 1)
        self.learning_rate=lr
        self.best_val_loss= float('inf')
        #self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4,batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                   batch_first=True)
        self.transformer= nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.activation= torch.nn.LeakyReLU(5e-4)
        self.fc1 = nn.Linear(200, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x):
        #x=self.embedding(x.long())
        out = self.transformer(x)
        out = out.view(out.shape[0],-1)
        out = self.activation(self.fc1(out))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))

        out = torch.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets.view(-1, 1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        if(loss<self.best_val_loss== float('inf')):
            self.best_val_loss=loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_loss = nn.MSELoss()(outputs, targets.view(-1, 1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        test_loss = nn.MSELoss()(outputs, targets.view(-1, 1))
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.8,
                                                              patience=500)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'monitor': 'train_loss',
            }
        }

        #return optimizer

