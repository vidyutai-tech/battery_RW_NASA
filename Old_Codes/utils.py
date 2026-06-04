import matplotlib.pyplot as plt
import torch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def moving_average(input_tensor, window_size):
    batch_size, sequence_length, num_features = input_tensor.shape
    output_tensor = torch.zeros_like(input_tensor)
    
    for b in range(batch_size):
        for f in range(num_features):
            for t in range(sequence_length):
                start = max(0, t - window_size)
                end = min(sequence_length, t + window_size + 1)
                output_tensor[b, t, f] = input_tensor[b, start:end, f].mean()
    
    return output_tensor


def plot_one_sequence_data(original_action, original_next_state):
    # Extract features
    current = original_action.squeeze().numpy()
    voltage = original_next_state[:, 0].numpy()
    temperature = original_next_state[:, 1].numpy()

    # Create a figure with a specific layout
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # Plot voltage values in the first subplot (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(voltage, label='Voltage (V)', color='b')
    ax1.set_xlabel('Sequence Index')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage over Sequence')
    ax1.legend()
    ax1.grid(True)

    # Plot temperature values in the second subplot (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(temperature, label='Temperature (°C)', color='r')
    ax2.set_xlabel('Sequence Index')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Temperature over Sequence')
    ax2.legend()
    ax2.grid(True)

    # Plot current values in the third subplot (bottom, spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(current, label='Current (A)', color='g')
    ax3.set_xlabel('Sequence Index')
    ax3.set_ylabel('Current (A)')
    ax3.set_title('Current over Sequence')
    ax3.legend()
    ax3.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()


def transform_trapezium(tensor, lower_bound, upper_bound, constant_value, positive_slope, negative_slope):
    """
    Transforms a 3D PyTorch tensor based on the specified bounds, constant value, and slopes.
    Applies the transformation element-wise. Then plots the original tensor values against
    the transformed values.

    :param tensor: 3D PyTorch tensor to transform
    :param lower_bound: Lower bound for the transformation
    :param upper_bound: Upper bound for the transformation
    :param constant_value: Constant value to use within the bounds
    :param positive_slope: Slope for the line below the lower bound
    :param negative_slope: Slope for the line above the upper bound
    """
    # Apply transformation using vectorized operations
    lower_mask = tensor < lower_bound
    upper_mask = tensor > upper_bound
    within_bounds_mask = ~lower_mask & ~upper_mask

    transformed_tensor = torch.empty_like(tensor)
    transformed_tensor[lower_mask] = positive_slope * (tensor[lower_mask] - lower_bound) + constant_value
    transformed_tensor[upper_mask] = negative_slope * (tensor[upper_mask] - upper_bound) + constant_value
    transformed_tensor[within_bounds_mask] = constant_value

    # Flattening tensors for plotting
    # original_flat = tensor.flatten().numpy()
    # transformed_flat = transformed_tensor.flatten().numpy()

    # # Plotting
    # plt.scatter(original_flat, transformed_flat, alpha=0.5)
    # plt.xlabel('Original Tensor Values')
    # plt.ylabel('Transformed Tensor Values')
    # plt.title('3D Tensor Transformation with Custom Slopes')
    # plt.show()
    return transformed_tensor


def transform_constant_to_linear(tensor, threshold, decrease_slope):
    """
    Transforms a 3D PyTorch tensor such that the values remain constant up to a threshold,
    and decrease linearly beyond that threshold. Applies the transformation element-wise.
    Then plots the original tensor values against the transformed values.

    :param tensor: 3D PyTorch tensor to transform
    :param threshold: Threshold value for the transformation
    :param decrease_slope: Slope for the linear decrease beyond the threshold
    """
    # Apply transformation using vectorized operations
    below_threshold_mask = tensor <= threshold
    above_threshold_mask = tensor > threshold

    transformed_tensor = torch.empty_like(tensor)
    transformed_tensor[below_threshold_mask] = threshold
    transformed_tensor[above_threshold_mask] = - decrease_slope * (tensor[above_threshold_mask] - threshold)

    # Flattening tensors for plotting
    # original_flat = tensor.flatten().numpy()
    # transformed_flat = transformed_tensor.flatten().numpy()

    # # Plotting
    # plt.scatter(original_flat, transformed_flat, alpha=0.5)
    # plt.xlabel('Original Tensor Values')
    # plt.ylabel('Transformed Tensor Values')
    # plt.title('3D Tensor Transformation with Linear Decrease')
    # plt.show()
    return transformed_tensor

def transform_trapezium_and_linear(tensor, lower_bound, upper_bound, constant_value, positive_slope, negative_slope):
    """
    Transforms a 3D PyTorch tensor based on the specified bounds, constant value, and slopes.
    Applies the transformation element-wise. Then plots the original tensor values against
    the transformed values.

    :param tensor: 3D PyTorch tensor to transform
    :param lower_bound: Lower bound for the transformation
    :param upper_bound: Upper bound for the transformation
    :param constant_value: Constant value to use within the bounds
    :param positive_slope: Slope for the line below the lower bound
    :param negative_slope: Slope for the line above the upper bound
    """
    # Apply transformation using vectorized operations
    lower_mask = tensor < lower_bound
    upper_mask = tensor > upper_bound
    within_bounds_mask = ~lower_mask & ~upper_mask
    slope_within_bounds = (1.5-constant_value)/(upper_bound-lower_bound) 


    transformed_tensor = torch.empty_like(tensor)
    transformed_tensor[lower_mask] = positive_slope * (tensor[lower_mask] - lower_bound) + constant_value
    negetive_slope_2 = -(0.5/0.2)
    transformed_tensor[upper_mask] = negetive_slope_2 * (tensor[upper_mask] - upper_bound) + 1.5#+ (upper_bound-lower_bound)
    
    
    transformed_tensor[within_bounds_mask] = slope_within_bounds * (tensor[within_bounds_mask] - lower_bound) + constant_value
    # Flattening tensors for plotting
    # original_flat = tensor.flatten().numpy()
    # transformed_flat = transformed_tensor.flatten().numpy()

    # # Plotting
    # plt.scatter(original_flat, transformed_flat, alpha=0.5)
    # plt.xlabel('Original Tensor Values')
    # plt.ylabel('Transformed Tensor Values')
    # plt.title('3D Tensor Transformation with Custom Slopes')
    # plt.show()
    return transformed_tensor

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.input_layer = nn.Linear(1, 10)  # One input feature, 10 hidden units
        self.hidden_layer = nn.Linear(10, 1) # 10 hidden units, one output

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.sigmoid(self.hidden_layer(x))
        return x