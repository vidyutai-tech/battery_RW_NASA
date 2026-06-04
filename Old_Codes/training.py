from imports_libraries import *
from model import TransformerModel, Transformer_Model, TransformerMode_Sequential, TransformerModel3, LSTMModel
from model import rescaling_current_additive_Correct_TransformerModel3
from model import Correct_LSTMModel,Correct_TransformerModel3, rescaling_Correct_TransformerModel3
from model import rescaling2_current_additive_Correct_TransformerModel3
from model import rescaling2_current_additive_Correct_TransformerModel3Decoder
#from model import rescaling_adaptive_TransformerModel3Decoder
from model import v2_rescaling_adaptive_TransformerModel3Decoder
from model import v3_rescaling_adaptive_TransformerModel3Decoder
from model import v4_rescaling_adaptive_TransformerModel3Decoder
from model import v5_rescaling_adaptive_TransformerModel3Decoder
from model import v6_rescaling_adaptive_TransformerModel3Decoder
from model import v7_rescaling_adaptive_TransformerModel3Decoder
from model import v8_rescaling_adaptive_TransformerModel3Decoder
from model import v9_rescaling_adaptive_TransformerModel3Decoder
from torch.utils.data import DataLoader, TensorDataset
# from data_loading import train_state,train_action,train_next_state
# from data_loading import val_state,val_action,val_next_state

from data_loading import train_dataset,test_dataset,val_dataset
def calculate_dimensional_mape(target, prediction):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) for each of the three dimensions.

    Parameters:
    target (torch.Tensor): The target tensor with shape (x, y, 3)
    prediction (torch.Tensor): The prediction tensor with shape (x, y, 3)

    Returns:
    tuple: A tuple containing the MAPE for each of the three dimensions
    """
    assert target.shape == prediction.shape, "Target and prediction must have the same shape"
    
    # Avoid division by zero
    target, prediction = target + 1e-8, prediction + 1e-8

    # Calculate MAPE for each dimension
    mape1 = torch.mean(torch.abs((target[:, :, 0] - prediction[:, :, 0]) / target[:, :, 0])) * 100
    mape2 = torch.mean(torch.abs((target[:, :, 1] - prediction[:, :, 1]) / target[:, :, 1])) * 100
    #mape3 = torch.mean(torch.abs((target[:, :, 2] - prediction[:, :, 2]) / target[:, :, 2])) * 100
    mape3=-1


    return mape1.item(), mape2.item(), mape3#.item()

train_1_mape= [ ]
train_2_mape= [ ]
train_3_mape= [ ]

valid_1_mape= []
valid_2_mape= []
valid_3_mape= []
max_norm = 0.6
def train_model(model,  
                train_dataset,test_dataset,val_dataset,
                num_epochs=10, learning_rate=0.001, batch_size=48,
                device='cpu',
                  sequence_length=10, save_path='model.pth'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)


    # Create DataLoader for training data
    #train_dataset = TensorDataset(*train_data)
    train_dataset=train_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create DataLoader for validation data
    #val_dataset = TensorDataset(*val_data)
    val_dataset=val_dataset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    val_loss_min =np.inf
    train_loss_best =np.inf
    #j=0
    train_loss_epoch = []
    valid_loss_epoch = []
    for epoch in range(num_epochs):
        print(epoch)
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            state, action, next_state = batch
            state, action, next_state =state.to(device), action.to(device), next_state.to(device) 
            print(state.shape,action.shape,next_state.shape,"$$$$$$$$")
            print(jd)

            optimizer.zero_grad()
            #print(state.shape, action.shape,next_state.shape,"$$$$$$$$$$$")
            #state, action =state[:,10-j,:], action[:,10-j,:]
            output = model(state, action)

            
            # print(output)
            #print(output.shape,"#############")
            #j=j+1
            
            
            # print(next_state)
            # print("##########################")
            # print(output)
            # print("##########################")
            # print(next_state.shape,output.shape,"$$$$$$$$$$$$$$")
            # # print(output.shape,"### Output shape")
            # # print(state.shape,action.shape,"$$$$$$$$$$$")

            # print(jd)
            loss = criterion(output.clone().detach(), next_state.clone().detach())
            # print(loss,"!@#@!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(jd)

            loss_normalized = (100*criterion(output[:,:,0], next_state[:,:,0]))+criterion(output[:,:,1], next_state[:,:,1])
            # loss.backward()
            # print(loss_normalized)
            # print(jd)
            loss_normalized.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


            mape1,mape2,mape3 = calculate_dimensional_mape(target = next_state,
                                                           prediction=output)

            train_1_mape.append(mape1)
            train_2_mape.append(mape2)
            train_3_mape.append(mape3)
            np.save('train_1_mape.npy', train_1_mape)
            np.save('train_2_mape.npy', train_2_mape)
            np.save('train_3_mape.npy', train_3_mape)



            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                val_state, val_action, val_next_state = batch
                val_state, val_action, val_next_state =val_state.to(device), val_action.to(device), val_next_state.to(device) 
                val_output = model(val_state, val_action)
                val_loss += criterion(val_output, val_next_state).item()
                mape1,mape2,mape3 = calculate_dimensional_mape(target = val_next_state,
                                                           prediction=val_output)

                valid_1_mape.append(mape1)
                valid_2_mape.append(mape2)
                valid_3_mape.append(mape3)
                np.save('valid_1_mape.npy', valid_1_mape)
                np.save('valid_2_mape.npy', valid_2_mape)
                np.save('valid_3_mape.npy', valid_3_mape)
    

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_loss_epoch.append(train_loss)
        valid_loss_epoch.append(val_loss)
        # np.save('train_mape.npy', train_loss_epoch)
        # np.save('valid_1_mape.npy', valid_loss_epoch)
        if(val_loss<val_loss_min):
            torch.save(model.state_dict(), save_path)
            val_loss_min = val_loss
            train_loss_best = train_loss
            print(f'Model saved to {save_path}')
            print(f'Best Val loss is {val_loss_min} and corresponding train loss is {train_loss_best}')

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)

          
    # Save the model
    

# Example usage
batch_size = 64
sequence_length = 150
# Initialize random tensors for training and validation
# Create model
# model = TransformerModel(time_embedding_size=0,state_input_size=3,state_output_size=3,
#                          action_input_size=1, output_size=4, 
#                          sequence_length=sequence_length)
# model = Transformer_Model(state_input_size=3,action_input_size=1,output_size=3,
#                           sequence_length=10)

# model = TransformerMode_Sequential(time_embedding_size=0,state_input_size=3,state_output_size=3,
#                          action_input_size=1, output_size=4, 
#                          sequence_length=sequence_length)

# model = Correct_TransformerModel3(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)

# model = rescaling_Correct_TransformerModel3(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)

# model = rescaling_current_additive_Correct_TransformerModel3(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)
# model = rescaling2_current_additive_Correct_TransformerModel3(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)
# model = v2_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)

# model = v3_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=8,
#                           dropout=0.3)

# model = v4_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=9,num_layers=4,
#                           dropout=0.3) #
# path for above '/Users/paarthsachan/simulator/v3_hid32_4hd_l1_decoder_transformer_diff_rescaled_additive_3.pth'
# model = v4_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=146,output_dim=2,nhead=10,num_layers=2,
#                           dropout=0.3)

# model = v4_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=146,output_dim=2,nhead=5,num_layers=1,
#                           dropout=0.1)

# model = v5_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=146,output_dim=2,nhead=10,num_layers=1,
#                           dropout=0.1)

# model = v6_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=150,output_dim=2,nhead=10,num_layers=1,
#                           dropout=0.1)

# model = v8_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=150,output_dim=2,nhead=10,num_layers=1,
#                           dropout=0.1)

model = v9_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=150,output_dim=2,nhead=20,num_layers=1,
                          dropout=0.1)
## Path for above v4_hid32_4hd_l1_decoder_transformer_diff_rescaled_additive_3
##########################################################################################
### RELOADING MODEL FOR RETRAINING THE SAME MODEL
##########################################################################################
saved_model_path = '/Users/paarthsachan/simulator/v11_hid32_4hd_l1_decoder_transformer_diff_rescaled_additive_3.pth'

model_state_dict = torch.load(saved_model_path)
model.load_state_dict(model_state_dict)
############################################################################################################
# model = LSTMModel(input_size_state=3, input_size_action=1, 
#                   hidden_size=256, num_layer=4,output_size=2)

device = torch.device("mps")
# Train the model and save it

train_model(model,  
            train_dataset=train_dataset,test_dataset=test_dataset,val_dataset=val_dataset,
            batch_size=128, 
            device = device,num_epochs=10000,learning_rate=5e-7,
            sequence_length=sequence_length, save_path='v12_hid32_4hd_l1_decoder_transformer_diff_rescaled_additive_3.pth')
