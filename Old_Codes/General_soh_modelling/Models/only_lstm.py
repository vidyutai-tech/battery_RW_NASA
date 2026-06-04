import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR,CyclicLR


class Only_LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, class_to_range,hidden_2=150,lr=4e-3,
                                                            output_size=2,
                                                            is_selecting_arch=False):
        super(Only_LSTMModel, self).__init__()
        self.learning_rate=lr
        self.best_val_loss= float('inf')
        self.best_val_acc= float('inf')
        self.best_model_train_loss= float('inf')
        self.states_concatenated = hidden_2
        self.output_dim=output_size

        self.model_ = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #self.layer_norm=nn.InstanceNorm1d(200, affine=True)
        self.activation=torch.nn.LeakyReLU(5e-4)
        #self.dropout=nn.Dropout(p=0.3)
        self.transformation = nn.Linear(self.states_concatenated*hidden_size,self.states_concatenated)
        self.fc1 = nn.Linear(self.states_concatenated,120)
        self.fc2 = nn.Linear(120,100)
        self.fc3 = nn.Linear(100,90)
        self.final_layer = nn.Linear(90,output_size)
        self.is_selecting_arch=is_selecting_arch
        self.class_to_range =class_to_range

    def forward(self, x):
        #x = x.view(-1,10,4)
        out, _ = self.model_(x)
        #print(out.shape)
        #out = self.layer_norm(out)
        #out = out.reshape(out.shape[0],-1)
        out = out[:, -self.states_concatenated:, :]
        out = out.reshape(out.shape[0],-1)
        out=self.transformation(out)
        out = self.activation(out)
        out =self.activation(self.fc1(out))
        out =self.activation(self.fc2(out))
        out =self.activation(self.fc3(out))
        out=self.final_layer(out)
        out = 1.2*torch.sigmoid(out)

        return out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        target_class = targets["class"]
        target_values =  targets["bounds"]

        outputs = self(inputs)
        loss = nn.MSELoss()(outputs[0], target_values[0])+nn.MSELoss()(outputs[1], target_values[1])
        loss = loss/2
        # predictions = torch.argmax(outputs, dim=1)
        # accuracy = torch.sum(predictions == targets).item() / len(targets)
        
        self.log('train_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_acc', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        target_class = targets["class"]
        target_values =  targets["bounds"]

        outputs = self(inputs)
        val_loss =  nn.MSELoss()(outputs[0], target_values[0])+nn.MSELoss()(outputs[1], target_values[1])
        val_loss = val_loss/2
        #val_loss = nn.MSELoss()(outputs, targets.view(targets.shape[0]*10,-1, 1))
        # predictions = torch.argmax(outputs, dim=1)
        # accuracy = torch.sum(predictions == targets).item() / len(targets)

        self.log('val_loss', val_loss,on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_acc', accuracy, on_epoch=True, prog_bar=True, logger=True)

        # if(val_loss<self.best_val_loss):
        #     self.best_val_loss=val_loss
        #     self.best_val_acc = accuracy
        #     if(self.is_selecting_arch==False):
        #         torch.save(self.state_dict()
        #         , '/Users/paarthsachan/technical/State_of_health_battery/General_soh_modelling/best_model/best_model_checkpoint.pth')
        return val_loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        target_class = targets["class"]
        target_values =  targets["bounds"]

        outputs = self(inputs)
        test_loss =  nn.MSELoss()(outputs[0], target_values[0])+nn.MSELoss()(outputs[1], target_values[1])
        test_loss = test_loss/2
        # predictions = torch.argmax(outputs, dim=1)
        # accuracy = torch.sum(predictions == targets).item() / len(targets)

        self.log('test_loss', test_loss,on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_acc', accuracy, on_epoch=True, prog_bar=True, logger=True)

        return test_loss


    def configure_optimizers(self):
        
        optimizer=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.95,
                                                              patience=500)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'monitor': 'val_loss',
            }
        }
    


