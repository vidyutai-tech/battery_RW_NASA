from imports_libraries import *
from scipy.io import loadmat
from torch.utils.data import Dataset, random_split
import numba

path_9='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW9.mat'
path_10='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW10.mat'
path_11='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW11.mat'
path_12='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW12.mat'
number_of_partial_profiles=1
subarray_length=150
annots = loadmat(path_9)
annots_=annots['data'][0][0]
steps=annots_[0][0]


def create_tensor_array(element, num_digits):
    place_values = 10**torch.arange(num_digits - 1, -1, -1)
    digits = (element // place_values).fmod(10) / 10
    return digits



last_reference_discharge=0
reference_discharge_indexes=[]
list_=[]
for i in range(len(steps)):
    step=steps[i]
    #print(step[0][0])
    list_.append(i)
    type_=step[0][0]
    if(type_=='reference discharge'):
        last_reference_discharge=i
        reference_discharge_indexes.append(i)

Voltage_array_=[]
Current_array_=[]
Time_array_=[]
Temperature_array_=[]

non_relative_time_stitched =[]
relative_time_stitched= []
current_stitched= []
voltage_stitched= []
temperature_stitched= []
age_stitched = []

for i in range(len(list_)):
    step=steps[i]
    #print(step[0][0])
    type_=step[0][0]
    relative_time_array=step[3][0]

    non_relative_time_array=step[2][0]
    voltage_array=step[4][0]
    current_array=step[5][0]
    temperature_array=step[6][0]
    age_array = np.ones(temperature_array.shape)
    age_array = (i/len(list_))*age_array


    ## time and non relative time are opposite of their names here
    #stitching procedure
    #print(voltage_array.shape)
    voltage_stitched.extend(voltage_array)
    current_stitched.extend(current_array)
    relative_time_stitched.extend(relative_time_array)
    non_relative_time_stitched.extend(non_relative_time_array)
    temperature_stitched.extend(temperature_array)
    age_stitched.extend(age_array)


    
    
    


print(len(list_),"#############")
# print(max(non_relative_time_stitched),print(max(relative_time_stitched)))

num_non_rel_time=len(str(max(non_relative_time_stitched)))
num_rel_time=len(str(max(relative_time_stitched)))
#9 and 11 here 
class MyDataset(Dataset):
    def __init__(self, non_relative_time_stitched, relative_time_stitched, 
                 current_stitched, voltage_stitched, temperature_stitched,
                 age_stitched,
                 chunk_size=150):
        self.non_relative_time_stitched = torch.tensor(non_relative_time_stitched,dtype=torch.float32)
        self.relative_time_stitched = torch.tensor(relative_time_stitched, dtype=torch.float32)
        self.current_stitched = torch.tensor(current_stitched, dtype=torch.float32)
        self.voltage_stitched = torch.tensor(voltage_stitched, dtype=torch.float32)
        self.temperature_stitched = torch.tensor(temperature_stitched, dtype=torch.float32)
        self.age_stitched = torch.tensor(age_stitched, dtype=torch.float32)

        self.temperature_stitched=torch.unsqueeze(self.temperature_stitched,dim=1)
        self.voltage_stitched=torch.unsqueeze(self.voltage_stitched,dim=1)
        self.current_stitched=torch.unsqueeze(self.current_stitched,dim=1)
        self.age_stitched = torch.unsqueeze(self.age_stitched,dim=1)

        # Assuming all arrays have the same length
        self.length = len(non_relative_time_stitched)
        self.chunk_size=chunk_size
        self.number_chunks = self.length//self.chunk_size

    def __len__(self):
        return self.number_chunks-1

    def __getitem__(self, index):
        starting_index =  self.chunk_size*index
        ending_index = starting_index+self.chunk_size+1
        non_relative_transformed = create_tensor_array(self.non_relative_time_stitched[starting_index],
                                                       num_non_rel_time)
        #non_relative_transformed=torch.tensor(non_relative_transformed)
        starting_temperature = self.temperature_stitched[starting_index]
        starting_voltage = self.voltage_stitched[starting_index]
        starting_time = self.age_stitched[starting_index]
        
        #print(non_relative_transformed.shape,starting_temperature.shape)
        starting_state = torch.cat([starting_time,starting_voltage, starting_temperature], dim=0)
        #print(starting_state.shape)
        next_temperatures = self.temperature_stitched[starting_index+1:ending_index]
        next_voltages = self.voltage_stitched[starting_index+1:ending_index]
        next_time = self.age_stitched[starting_index+1:ending_index]
        #print(next_temperatures.shape, next_voltages.shape)
        next_states = torch.cat([next_voltages,next_temperatures], dim=1)

        actions = self.current_stitched[starting_index+1:ending_index]
        
        return starting_state ,actions,next_states
    
dataset = MyDataset(non_relative_time_stitched=non_relative_time_stitched,
                    relative_time_stitched=relative_time_stitched,
                    current_stitched=current_stitched,
                    voltage_stitched=voltage_stitched,
                    temperature_stitched=temperature_stitched,
                    age_stitched=age_stitched)

torch.manual_seed(42)
#### original experimemts have manual seed 42
total_size = len(dataset)
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.1

train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

indices = torch.randperm(total_size)

train_indices = indices[:train_size]
val_indices = indices[train_size:(train_size + val_size)]
test_indices = indices[(train_size + val_size):]

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# # Example usage
# batch_size = 16
# sequence_length = 10
# data_size =32
# # Initialize random tensors for training and validation


# train_state = torch.rand((data_size, 12))## 11+ 1(temperature)
# train_action = torch.rand((data_size, sequence_length, 1))
# train_next_state = torch.rand((data_size, sequence_length, 3))

# # val_state = torch.rand((data_size, 3))
# # val_action = torch.rand((data_size, sequence_length, 1))
# # val_next_state = torch.rand((data_size, sequence_length, 3))

# val_state = train_state
# val_action = train_action
# val_next_state =train_next_state


# #print(train_state)
# #print(jd)