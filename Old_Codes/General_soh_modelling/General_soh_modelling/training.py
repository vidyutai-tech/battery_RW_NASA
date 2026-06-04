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

checkpoint_callback = callbacks.ModelCheckpoint(
   monitor='train_loss',  # Monitor validation loss
    mode='min',          # Minimize validation loss
    save_top_k=1,        # Save the top 1 best model
    dirpath='checkpoints',  # Directory to save checkpoints
    filename='best_model',  # Checkpoint filename
    save_last=False

)

class SaveLastModelCallback(pl.Callback):
    def __init__(self, threshold):
        self.threshold = threshold

    def on_epoch_end(self, trainer, pl_module):
        # Check if the training loss is below the specified threshold
        if pl_module.trainer.callback_metrics['train_loss'] <= self.threshold:
            # Save the last model
            trainer.save_checkpoint("last_model.ckpt")
save_last_model_callback = SaveLastModelCallback(threshold=9e-4)


early_stopping_callback = EarlyStopping(monitor='train_loss', min_delta=9e-4, patience=3, verbose=True, mode='min')

logger_tensorboard = TensorBoardLogger(save_dir="logs", name="my_lstm_model_equal_data_split") 
class_to_range=data_module.class_to_range

def objective(trial):
    # Create a new instance of the model with suggested hyperparameters
    hidden_suggest = trial.suggest_int('hidden_suggest',8,8,16)
    num_layers_suggest = trial.suggest_int('num_layers_suggest',1,2,1)
    lr_suggest = trial.suggest_loguniform('lr_suggest', 1e-5, 1e-3)
    model = Only_LSTMModel(
        input_size=3,
        hidden_size=hidden_suggest,
        num_layers=num_layers_suggest,
        class_to_range=class_to_range,
        output_size=2,
        lr=lr_suggest,
        is_selecting_arch=True
    )
    # Define the Lightning trainer with appropriate callbacks
    trainer = pl.Trainer(
        logger=True,
        max_epochs=5000,
        accelerator="auto",
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_loss')],
        deterministic=True
    )

    hyperparameters = dict(hidden=hidden_suggest, layers= num_layers_suggest,lr=lr_suggest)
    trainer.logger.log_hyperparams(hyperparameters)


    # Fit the model
    trainer.fit(model, datamodule=data_module)

    # Get the validation loss from the best model checkpoint
    #checkpoint = torch.load(checkpoint_callback.best_model_path)
    val_loss = model.best_val_loss

    return val_loss


study = optuna.create_study(direction='minimize',pruner=optuna.pruners.NopPruner()
                            ,sampler=TPESampler())  # We want to minimize the validation loss

# Create a PyTorch Lightning callback for pruning


# Run the optimization
study.optimize(objective, n_trials=4)
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_hyperparameters = study.best_params
hidden_size = best_hyperparameters['hidden_suggest']
num_layers = best_hyperparameters['num_layers_suggest']
learning_rate = best_hyperparameters['lr_suggest']

best_model = Only_LSTMModel(
    input_size=3,
    hidden_size=hidden_size,
    class_to_range=class_to_range,
    num_layers=num_layers,
    output_size=2,
    lr=learning_rate
)

trainer_ = pl.Trainer(
    logger=logger_tensorboard,
    max_epochs=30000,
    accelerator="auto",
    callbacks=[checkpoint_callback],
    deterministic=True
)
print("Best hyperarams params are:", hidden_size,num_layers)
trainer_.fit(best_model, datamodule=data_module)
trainer_.test(best_model, datamodule=data_module)
path=checkpoint_callback.best_model_path
print(path,"##############")
# path ='/Users/paarthsachan/technical/State_of_health_battery/General_soh_modelling/best_model/best_model_checkpoint.pth'
# best_model_trained = Only_LSTMModel.load_from_checkpoint(path,
#                         input_size=3,  hidden_size=hidden_size,
#                        class_to_range=class_to_range, num_layers=num_layers, 
#                         output_size=2)

# trainer_.test(best_model, datamodule=data_module)

# print("Best model loaded")

# val_dataloader= data_module.val_dataloader()
# train_dataloader= data_module.train_dataloader()
# test_dataloader = data_module.test_dataloader()
# train_loss = 0.0
# val_loss = 0.0
# test_loss = 0.0
# mse_loss = nn.MSELoss()

# # Set the model to evaluation mode
# model = best_model
# model.eval()

# # Calculate the MSE loss for the training set
# with torch.no_grad():
#     loss=0
#     for batch in train_dataloader:
#         inputs, targets = batch
#         target_class = targets["class"]
#         target_values =  targets["bounds"]

#         outputs = model.forward(inputs)
#         loss = nn.MSELoss()(outputs[0], target_values[0])+nn.MSELoss()(outputs[1], target_values[1])

# print(loss,"Train_loss")
# # Calculate the MSE loss for the validation set
# with torch.no_grad():
#     loss=0
#     for batch in val_dataloader:
#         inputs, targets = batch
#         target_class = targets["class"]
#         target_values =  targets["bounds"]

#         outputs = model.forward(inputs)
#         loss = nn.MSELoss()(outputs[0], target_values[0])+nn.MSELoss()(outputs[1], target_values[1])

# print(loss,"Val_loss")
# with torch.no_grad():
#     loss=0
#     for batch in test_dataloader:
#         inputs, targets = batch
#         target_class = targets["class"]
#         target_values =  targets["bounds"]

#         outputs = model.forward(inputs)
#         loss = nn.MSELoss()(outputs[0], target_values[0])+nn.MSELoss()(outputs[1], target_values[1])

# print(loss,"Test losss")



# # print(f"CE Loss for Training Set for Best Model: {train_loss}")
# # print(f"CE Loss for Validation Set for Best Model: {val_loss}")
# print(path,"##############")