
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from data_loading import data_module
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, callbacks
from Models.lstm_model import LSTMModel
from Models.tranformer import TransformerModel
from Models.only_lstm import Only_LSTMModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna import trial
from optuna.samplers import TPESampler
import copy

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device= torch.device("cpu")

class_to_range=data_module.class_to_range
path ='/Users/paarthsachan/technical/State_of_health_battery/General_soh_modelling/checkpoints/best_model-v33.ckpt'
# path ='/Users/paarthsachan/technical/State_of_health_battery/General_soh_modelling/best_model/best_model_checkpoint.pth'
best_model_trained = Only_LSTMModel.load_from_checkpoint(path,
                        input_size=3,  hidden_size=8,
                       class_to_range=class_to_range, num_layers=1, 
                        output_size=2)


print("Best model loaded")

val_dataloader= data_module.val_dataloader()
train_dataloader= data_module.train_dataloader()
test_dataloader = data_module.test_dataloader()
train_loss = 0.0
val_loss = 0.0
test_loss = 0.0
mse_loss = nn.MSELoss()

# Set the model to evaluation mode
model = best_model_trained
model.eval()

# Calculate the MSE loss for the training set
with torch.no_grad():
    loss=0
    for batch in train_dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        target_class = targets["class"].to(device)
        target_values =  targets["bounds"].to(device)

        outputs = model.forward(inputs)
        loss += nn.MSELoss()(outputs[0], target_values[0])+nn.MSELoss()(outputs[1], target_values[1])
        
print(loss/len(train_dataloader),"Train_loss")
# Calculate the MSE loss for the validation set
with torch.no_grad():
    loss=0
    for batch in val_dataloader:
        inputs, targets = batch
        inputs=inputs.to(device)
        target_class = targets["class"].to(device)
        target_values =  targets["bounds"].to(device)

        outputs = model.forward(inputs)
        loss = nn.MSELoss()(outputs[0], target_values[0])+nn.MSELoss()(outputs[1], target_values[1])

print(loss,"Val_loss")
with torch.no_grad():
    loss=0
    for batch in test_dataloader:
        inputs, targets = batch
        inputs=inputs.to(device)
        target_class = targets["class"].to(device)
        target_values =  targets["bounds"].to(device)
        
        outputs = model.forward(inputs)
        print(outputs.shape)
        print(inputs.shape)
        print(inputs)
        break
        loss += nn.MSELoss()(outputs[0], target_values[0])+nn.MSELoss()(outputs[1], target_values[1])

print(loss/len(test_dataloader),"Test losss")


# print(f"CE Loss for Training Set for Best Model: {train_loss}")
# print(f"CE Loss for Validation Set for Best Model: {val_loss}")
print(path,"##############")