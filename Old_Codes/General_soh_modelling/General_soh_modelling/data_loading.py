import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.io import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, Subset
np.random.seed(42)
from torch.utils.data.dataset import random_split


path_9='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW9.mat'
path_10='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW10.mat'
path_11='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW11.mat'
path_12='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW12.mat'
number_of_partial_profiles=1
subarray_length=150
annots = loadmat(path_9)
annots_=annots['data'][0][0]
steps=annots_[0][0]


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

def find_unique_elements(arr):
    unique_elements = set(arr)
    return list(unique_elements)



def carve_array_into_subarrays(input_array, number_of_partial_profiles=number_of_partial_profiles
                               ,subarray_length=subarray_length):
    if number_of_partial_profiles <= 0:
        raise ValueError("Number of partial profiles must be greater than 0")

    # Calculate the length of each subarray
    

    # Initialize an empty list to store the subarrays
    skip_between_two_arr=((len(input_array)-
                  subarray_length*number_of_partial_profiles)//number_of_partial_profiles)-1
    subarrays = []

    # Loop to create the subarrays
    for i in range(number_of_partial_profiles):
        start_index = i * subarray_length
        end_index = (i + 1) * subarray_length

        # Append the subarray to the list
        if(i!=0):
            start_index+=i*skip_between_two_arr
            end_index+=i*skip_between_two_arr
        subarrays.append(input_array[start_index:end_index])

    return subarrays


integral_I_dt=[]
Voltage_array_=[]
Current_array_=[]
Time_array_=[]
Temperature_array_=[]
lower_bound_soh = []
upper_bound_soh =[]
charging_index=[]
for index_current in tqdm(range(len(list_))):
    index_current_=list_[index_current]
    step=steps[index_current_]
    type_=step[0][0]
    time_array=step[3][0]

    non_relative_time=step[2][0]
    voltage_array=step[4][0]#+1e-7
    current_array=step[5][0]#+1e-7
    temperature_array=step[6][0]

    if((type_=='reference charge')or(type_=='charge (random walk)')):
        charging_index.append(index_current_)
        #print(type_)
    
    #indexes_distributed=evenly_distributed_indexes(voltage_array)
    Voltage_array_.extend(carve_array_into_subarrays(voltage_array))
    Current_array_.extend(carve_array_into_subarrays(current_array))
    Time_array_.extend(carve_array_into_subarrays(time_array))
    Temperature_array_.extend(carve_array_into_subarrays(temperature_array))
    
    if(type_=='reference discharge'):
        I_dt = np.trapz(current_array, x=time_array)
        integral_I_dt.append(I_dt/(2.2*3600))
    else:
        integral_I_dt.append(-1)


next_lower_bound=None
count=0
class_of_step=[] ### could be simply indexed to find the lower and upper bounds
print(len(integral_I_dt))
for i in range(len(integral_I_dt)):
    if(i<reference_discharge_indexes[0]):
        lower_bound_soh.append(integral_I_dt[reference_discharge_indexes[0]])
        upper_bound_soh.append(1)
    else:
        if((integral_I_dt[i] == -1)and (count!=80)):
            lower_bound_soh.append(integral_I_dt[reference_discharge_indexes[count]])
            upper_bound_soh.append(integral_I_dt[reference_discharge_indexes[count-1]])

        elif((count!=80)):
            count+=1
            if(count!=80):
                lower_bound_soh.append(integral_I_dt[reference_discharge_indexes[count]])
                upper_bound_soh.append(integral_I_dt[reference_discharge_indexes[count-1]])
            else:
                lower_bound_soh.append(0)
                upper_bound_soh.append(integral_I_dt[reference_discharge_indexes[-1]])

        else:
            lower_bound_soh.append(0)
            upper_bound_soh.append(integral_I_dt[reference_discharge_indexes[-1]])
    
    class_of_step.append(count)

unique_lower_bound = list(set(lower_bound_soh))
unique_upper_bound = list(set(upper_bound_soh))
class_to_range={}

unique_lower_bound.sort(reverse=True)
unique_upper_bound.sort(reverse=True)
for i in range(len(unique_lower_bound)):
    class_to_range[i]=[unique_lower_bound[i],unique_upper_bound[i]]



window_size=5
relevant_indicies=[]
for i in range(len(reference_discharge_indexes)):
    start_idx = reference_discharge_indexes[i]-window_size
    end_idx = reference_discharge_indexes[i] + window_size
    array_of_relevant = list_[start_idx:end_idx]
    relevant_indicies.extend(array_of_relevant)

temp_relevant_indicies= []
type_of_relevant= []
for i in range(len(relevant_indicies)):
    global_ind=relevant_indicies[i]
    step=steps[global_ind]
    type_=step[0][0]
    time_array=step[3][0]
    if(len(time_array)>=subarray_length):
        temp_relevant_indicies.append(global_ind)
        type_of_relevant.append(type_)

relevant_indicies=temp_relevant_indicies
relevant_indicies = list(set(relevant_indicies) & set(charging_index))



class CustomDataset(Dataset):
    def __init__(self, voltage_array, current_array, time_array, 
                 temperature_array, lower_bound_soh, 
                 upper_bound_soh, class_of_step, 
                 window_size,
                  relevant_indicies ):
        self.window_size = window_size
        self.data = {
            'voltage': voltage_array,
            'current': current_array,
            'time': time_array,
            'temperature': temperature_array,
            'lower_bound_soh': lower_bound_soh,
            'upper_bound_soh': upper_bound_soh,
            'class_of_step': class_of_step
        }

        self.relevant_indicies = relevant_indicies

    def __len__(self):
        return len(self.relevant_indicies)

    def __getitem__(self, idx):
        index_global = self.relevant_indicies[idx]

        voltage_tensor = torch.tensor(self.data['voltage'][index_global], dtype=torch.float32)
        current_tensor = torch.tensor(self.data['current'][index_global], dtype=torch.float32)
        temperature_tensor = torch.tensor(self.data['temperature'][index_global], dtype=torch.float32)
        
        upper_bound_tensor = torch.tensor(self.data['lower_bound_soh'][index_global], dtype=torch.float32)
        lower_bound_tensor = torch.tensor(self.data['upper_bound_soh'][index_global], dtype=torch.float32)
        bound_=torch.stack([lower_bound_tensor, upper_bound_tensor])
        
        class_of_step_tensor = torch.tensor(self.data['class_of_step'][index_global], dtype=torch.int32)
        
        model_input = torch.stack((voltage_tensor,current_tensor,temperature_tensor),dim=1)
        target = {"class":class_of_step_tensor,"bounds":bound_}
       
        return model_input, target


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, voltage_array, current_array, time_array, temperature_array, 
                 lower_bound_soh, upper_bound_soh, class_of_step, window_size, 
                 relevant_indices, batch_size=32, random_seed=42,class_to_range=class_to_range):
        super().__init__()
        self.dataset = CustomDataset(voltage_array, current_array, time_array, 
                                     temperature_array, lower_bound_soh, upper_bound_soh, 
                                     class_of_step, window_size, relevant_indices)
        self.batch_size = len(self.dataset)
        self.split_fractions = [0.7, 0.15, 0.15]
        self.random_seed = random_seed
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_data()
        self.class_to_range=class_to_range
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        #return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        #return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        #return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def _split_data(self):
        train_indicies=[]
        validation_indicies=[]
        testing_indicies=[]

        for i in range(len(self.dataset)):
            if((i+1)%5==3):
                validation_indicies.append(i)
            elif((i+1)%5==4):
                testing_indicies.append(i)
            else:
                train_indicies.append(i)
        
        train_data=Subset(self.dataset,train_indicies)
        val_data=Subset(self.dataset,validation_indicies)
        test_data=Subset(self.dataset,testing_indicies)

        return train_data, val_data, test_data
    
voltage_array = Voltage_array_
current_array = Current_array_
time_array = Time_array_
temperature_array = Temperature_array_
lower_bound_soh = lower_bound_soh
upper_bound_soh = upper_bound_soh
class_of_step = class_of_step

# Define your window size
window_size = 5

# Create the data module
data_module = CustomDataModule(
    voltage_array, current_array, time_array, temperature_array, lower_bound_soh, upper_bound_soh, class_of_step, window_size,
    relevant_indicies,batch_size=len(relevant_indicies)
)
print(data_module)
print(len(data_module.dataset))
# print(jd)
# are_all_numbers_present = all(i in class_of_step for i in range(1, 82))

# if are_all_numbers_present:
#     print("All numbers from 1 to 81 are present.")
# else:
#     print("Some numbers from 1 to 81 are missing.")


# missing_numbers = [i for i in range(1, 82) if i not in class_of_step]

# if not missing_numbers:
#     print("All numbers from 1 to 81 are present.")
# else:
#     print("Missing numbers:", missing_numbers)
## MINOR TESTING TO CHECK NUMBER OF CLASSES 

