import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR,CyclicLR


class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size,lr=4e-3):
        super(LSTMModel, self).__init__()
        self.learning_rate=lr
        self.best_val_loss= float('inf')
        self.best_model_train_loss= float('inf')

        self.model_ = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #self.layer_norm=nn.InstanceNorm1d(200, affine=True)
        self.activation=torch.nn.LeakyReLU(5e-4)
        #self.dropout=nn.Dropout(p=0.3)
        self.fc=nn.Linear(100*hidden_size, 2048)
        self.fc2=nn.Linear(2048, 512)
        self.fc3=nn.Linear(512, 64)
        self.fc4= nn.Linear(64,1)

    def forward(self, x):
        #x = x.view(-1,10,4)
        out, _ = self.model_(x)
        #print(out.shape)
        #out = self.layer_norm(out)
        out = out.reshape(out.shape[0],-1)

        #out = self.fc(out[:, -1, :])
        out = self.activation(self.fc(out))
        
        #out = self.dropout(out)
        out = self.activation(self.fc2(out))
        #out = self.dropout(out)
        out = self.activation(self.fc3(out))
        #out = self.dropout(out)

        out = self.fc4(out)

        out = torch.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets.view(-1, 1))
        #loss = nn.MSELoss()(outputs, targets.view(targets.shape[0]*10,-1, 1))
        
        
        self.log('train_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_loss = nn.MSELoss()(outputs, targets.view(-1, 1))
        #val_loss = nn.MSELoss()(outputs, targets.view(targets.shape[0]*10,-1, 1))
        self.log('val_loss', val_loss,on_epoch=True, prog_bar=True, logger=True)
        if(val_loss<self.best_val_loss):
            self.best_val_loss=val_loss
        return val_loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        test_loss = nn.MSELoss()(outputs, targets.view(-1, 1))
        #test_loss = nn.MSELoss()(outputs, targets.view(targets.shape[0]*10,-1, 1))
        self.log('test_loss', test_loss,on_epoch=True, prog_bar=True, logger=True)
        return test_loss


    def configure_optimizers(self):
        
        optimizer=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.4,
                                                              patience=500)

        # scheduler = CyclicLR(
        #     optimizer,
        #     base_lr=1e-8,  ##for cycle lr
        #     max_lr=4e-2  # Maximum learning rate
        #     #total_steps=50000,  # Total number of training steps (adjust as needed)
        #     #pct_start=0.1,  # Percentage of the cycle where the learning rate is increasing
        #     #anneal_strategy='cos',  # Annealing strategy ('cos' for cosine annealing)
        # )
        

        # # Use LearningRateMonitor callback to log learning rates
        # lr_monitor = LearningRateMonitor(logging_interval='step')

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'monitor': 'val_loss',
            }
        }
    
        #return [optimizer], [scheduler]

        #return optimizer



