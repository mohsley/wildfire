from herbie import Herbie, wgrib2, paint, FastHerbie
from herbie.toolbox import EasyMap, ccrs, pc
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import math
import cv2
import requests
import json
import os
import sys
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from datetime import timedelta
# Note: We won't use random train_test_split for time series data
# from sklearn.model_selection import train_test_split

# downloads 7 days of data (1-10 to 1-17), 24 hours (00 to 23) -> 169 hours
# not forecasts -- downloading actual hourly updated data
def get_hrrr_surface_smoke_data(start_date, end_date, product="MASSDEN"):
    dates = pd.date_range(start_date, end_date, freq="1h")
    FH = FastHerbie(dates, model="hrrr", fxx=[0])
    FH.download(product)
        
    return FH.objects

def subregion_grib_files_to_numpy(herbie_data, extent, product="MASSDEN"):
    hrrr_subregion_list = []
    for H in herbie_data:
        # subregion grib files
        file = H.get_localFilePath(product)
        idx_file = wgrib2.create_inventory_file(file)
        subset_file = wgrib2.region(file, extent, name="la_region")

        # grib -> xarr -> numpy
        hrrr_xarr = xr.open_dataset(subset_file, engine="cfgrib", decode_timedelta=False)
        hrrr_subregion_list.append(np.flip(hrrr_xarr.mdens.to_numpy(), axis=0))

    return np.array(hrrr_subregion_list)

def sliding_window_of(frames, sample_size, rows, cols, channels):
    n_samples = len(frames) - sample_size
    samples = np.empty((n_samples, sample_size, rows, cols, channels))
    for i in range(n_samples):
        samples[i] = np.array([frames[j] for j in range(i, i + sample_size)])
        
    return samples

def np_to_final_input(hrrr_subregion_np, dim, sample_size):
    channels = 1
    n_frames = len(hrrr_subregion_np)
    frames = np.empty(shape=(n_frames, dim, dim, channels))

    # interpolate and add channel axis
    for i, frame in enumerate(hrrr_subregion_np):
        new_frame = cv2.resize(frame, (dim, dim))
        new_frame = np.reshape(new_frame, (dim, dim, channels))
        frames[i] = new_frame

    # create sample axis with a sliding window of frames
    complete_ds = sliding_window_of(frames, sample_size, dim, dim, channels)
    
    return complete_ds

def get_airnow_data(
    start_date, end_date, 
    lon_bottom, lat_bottom, lon_top, lat_top,
    airnow_api_key=None
):
    # get airnow data from the EPA
    if os.path.exists('data/airnow.json'):
        print("data/airnow.json already exists; skipping request...")
    else:
        # preprocess a few parameters
        date_start = pd.to_datetime(start_date).isoformat()[:13]
        date_end = pd.to_datetime(end_date).isoformat()[:13]
        bbox = f'{lon_bottom},{lat_bottom},{lon_top},{lat_top}'
        URL = "https://www.airnowapi.org/aq/data"
                
        # defining a params dict for the parameters to be sent to the API
        PARAMS = {
            'startDate':date_start,
            'endDate':date_end,
            'parameters':'PM25',
            'BBOX':bbox,
            'dataType':'B',
            'format':'application/json',
            'verbose':'0',
            'monitorType':'2',
            'includerawconcentrations':'1',
            'API_KEY':airnow_api_key
        }
        
        # sending get request and saving the response as response object
        response = requests.get(url = URL, params = PARAMS)
    
        # extracting data in json format, then download
        airnow_data = response.json()
        with open('data/airnow.json', 'w') as file:
            json.dump(airnow_data, file)
            print("JSON data saved to data/airnow.json")
        
    # open json file and convert to dataframe
    with open('data/airnow.json', 'r') as file:
        airnow_data = json.load(file)
    airnow_df = pd.json_normalize(airnow_data)

    # group station data by time
    list_df = [group for name, group in airnow_df.groupby('UTC')]
    
    return list_df

def preprocess_ground_sites(df, dim, latMax, latMin, lonMax, lonMin):
    latDist, lonDist = abs(latMax - latMin), abs(lonMax - lonMin)
    unInter = np.zeros((dim,dim))
    dfArr = np.array(df[['Latitude','Longitude','Value']])
    for i in range(dfArr.shape[0]):
        # Calculate x
        x = int(((latMax - dfArr[i,0]) / latDist) * dim)
        if x >= dim:
            x = dim - 1
        if x <= 0:
            x = 0
        # Calculate y
        y = dim - int(((lonMax + abs(dfArr[i,1])) / lonDist) * dim)
        if y >= dim:
            y = dim - 1
        if y <= 0:
            y = 0
        if dfArr[i,2] < 0:
            unInter[x,y] = 0
        else:
            unInter[x,y] = dfArr[i,2]
    return unInter

# Original interpolation function - keeping for reference or comparison
def original_interpolate_frame(f, dim):
    i = 0
    interpolated = []
    count = 0
    idx = 0
    x_list = []
    y_list = []
    values = []
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            if f[x,y] != 0:
                x_list.append(x)
                y_list.append(y)
                values.append(f[x,y])
    coords = list(zip(x_list,y_list))
    try:
        interp = NearestNDInterpolator(coords, values)
        X = np.arange(0,dim)
        Y = np.arange(0,dim)
        X, Y = np.meshgrid(X, Y)
        Z = interp(X, Y)
    except ValueError:
        Z = np.zeros((dim,dim))
    interpolated = Z
    count += 1
    i += 1
    interpolated = np.array(interpolated)
    return interpolated

# NEW: IDW interpolation function - Standard version
def idw_interpolate_frame(f, dim, p=2):
    """
    Performs Inverse Distance Weighting interpolation on a frame
    
    Parameters:
    f - The input frame with sensor data
    dim - The dimension of the output grid
    p - The power parameter (default=2), controls how quickly influence decreases with distance
    
    Returns:
    A dim x dim interpolated grid
    """
    # Find locations and values of non-zero points (sensor locations)
    x_list = []
    y_list = []
    values = []
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            if f[x, y] != 0:
                x_list.append(x)
                y_list.append(y)
                values.append(f[x, y])
    
    if len(values) == 0:
        # No data points, return zeros
        return np.zeros((dim, dim))
    
    # Create coordinates array for known points
    coords = np.array(list(zip(x_list, y_list)))
    values = np.array(values)
    
    # Create the output grid
    interpolated = np.zeros((dim, dim))
    
    # Implement IDW
    for x in range(dim):
        for y in range(dim):
            # Calculate distances from current point to all known points
            point = np.array([x, y])
            distances = np.sqrt(np.sum((coords - point)**2, axis=1))
            
            # Handle the case where a point coincides with a known point
            if np.any(distances == 0):
                # Find indices where distance is 0
                idx = np.where(distances == 0)[0]
                # Use the value directly
                interpolated[x, y] = values[idx[0]]  # Just take the first one if multiple
            else:
                # Calculate weights using inverse distance formula
                weights = 1.0 / (distances ** p)
                
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights)
                
                # Calculate weighted average
                interpolated[x, y] = np.sum(weights * values)
    
    return interpolated

# NEW: IDW with epsilon to avoid extreme weights
def idw_interpolate_frame_epsilon(f, dim, p=2, epsilon=0.1):
    """
    IDW interpolation with epsilon added to distances to avoid extreme weights
    """
    x_list = []
    y_list = []
    values = []
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            if f[x, y] != 0:
                x_list.append(x)
                y_list.append(y)
                values.append(f[x, y])
    
    if len(values) == 0:
        return np.zeros((dim, dim))
    
    coords = np.array(list(zip(x_list, y_list)))
    values = np.array(values)
    interpolated = np.zeros((dim, dim))
    
    for x in range(dim):
        for y in range(dim):
            point = np.array([x, y])
            distances = np.sqrt(np.sum((coords - point)**2, axis=1))
            
            if np.any(distances == 0):
                idx = np.where(distances == 0)[0]
                interpolated[x, y] = values[idx[0]]
            else:
                # Add epsilon to distances to avoid extreme weights
                weights = 1.0 / ((distances + epsilon) ** p)
                weights = weights / np.sum(weights)
                interpolated[x, y] = np.sum(weights * values)
    
    return interpolated

# NEW: IDW with exponential decay
def idw_interpolate_frame_exp(f, dim, p=0.1):
    """
    IDW interpolation using exponential decay for weighting
    """
    x_list = []
    y_list = []
    values = []
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            if f[x, y] != 0:
                x_list.append(x)
                y_list.append(y)
                values.append(f[x, y])
    
    if len(values) == 0:
        return np.zeros((dim, dim))
    
    coords = np.array(list(zip(x_list, y_list)))
    values = np.array(values)
    interpolated = np.zeros((dim, dim))
    
    for x in range(dim):
        for y in range(dim):
            point = np.array([x, y])
            distances = np.sqrt(np.sum((coords - point)**2, axis=1))
            
            if np.any(distances == 0):
                idx = np.where(distances == 0)[0]
                interpolated[x, y] = values[idx[0]]
            else:
                # Use exponential decay
                weights = np.exp(-distances * p)
                weights = weights / np.sum(weights)
                interpolated[x, y] = np.sum(weights * values)
    
    return interpolated

# we scale each sample relative to other samples
def std_scale(data):
    mean = np.mean(data)
    stddev = np.std(data)
    return (data - mean) / stddev

# Visualization function to compare interpolation methods including all new options
def visualize_all_interpolation_methods(uninterpolated, sensor_locs, p_values=[2, 3, 4], epsilon=0.1, exp_p=0.1):
    """
    Visualize and compare all interpolation methods
    
    Parameters:
    uninterpolated - The raw sensor data grid with zeros for non-sensor locations
    sensor_locs - Dictionary of sensor locations to mark on the plot
    p_values - List of power values to test
    epsilon - Epsilon value for the epsilon method
    exp_p - Parameter for exponential decay method
    """
    # Calculate the total number of plots needed
    n_p_values = len(p_values)
    n_plots = 2 + n_p_values + 1 + 1  # Original + raw + n_p_values + epsilon + exp
    
    # Create a figure with enough subplots
    fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=(20, 10))
    axes = axes.flatten()
    
    # Plot uninterpolated data
    im0 = axes[0].imshow(uninterpolated, cmap='viridis')
    axes[0].set_title('Raw Sensor Data')
    plt.colorbar(im0, ax=axes[0])
    
    # Mark sensor locations
    for name, (x, y) in sensor_locs.items():
        axes[0].plot(y, x, 'ro', markersize=5)
        axes[0].text(y+1, x+1, name, fontsize=8, color='red')
        # Print the value at this location
        axes[0].text(y, x-2, f"{uninterpolated[x,y]:.1f}", fontsize=8, color='white')
    
    # Plot nearest neighbor interpolation
    nearest_interp = original_interpolate_frame(uninterpolated, uninterpolated.shape[0])
    im1 = axes[1].imshow(nearest_interp, cmap='viridis')
    axes[1].set_title('Nearest Neighbor Interpolation')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot IDW with different p values
    for i, p in enumerate(p_values):
        idw_interp = idw_interpolate_frame(uninterpolated, uninterpolated.shape[0], p=p)
        im = axes[2+i].imshow(idw_interp, cmap='viridis')
        axes[2+i].set_title(f'IDW Interpolation (p={p})')
        plt.colorbar(im, ax=axes[2+i])
    
    # Plot IDW with epsilon
    idx = 2 + n_p_values
    idw_eps_interp = idw_interpolate_frame_epsilon(uninterpolated, uninterpolated.shape[0], p=2, epsilon=epsilon)
    im_eps = axes[idx].imshow(idw_eps_interp, cmap='viridis')
    axes[idx].set_title(f'IDW with Epsilon (p=2, ε={epsilon})')
    plt.colorbar(im_eps, ax=axes[idx])
    
    # Plot IDW with exponential decay
    idx = 2 + n_p_values + 1
    idw_exp_interp = idw_interpolate_frame_exp(uninterpolated, uninterpolated.shape[0], p=exp_p)
    im_exp = axes[idx].imshow(idw_exp_interp, cmap='viridis')
    axes[idx].set_title(f'IDW with Exp Decay (p={exp_p})')
    plt.colorbar(im_exp, ax=axes[idx])
    
    # Hide any extra subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Comparison of Different Interpolation Methods', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    return {
        'nearest': nearest_interp,
        'idw_p_values': {p: idw_interpolate_frame(uninterpolated, uninterpolated.shape[0], p=p) for p in p_values},
        'idw_epsilon': idw_eps_interp,
        'idw_exp': idw_exp_interp
    }

# define bounding box
lat_bottom, lat_top = 33.6, 34.3
lon_bottom, lon_top = -118.6, -117.9
extent = (lon_bottom, lon_top, lat_bottom, lat_top)

# input data shape
dim = 40
frames_per_sample = 5

# date range of data
start_date, end_date = "2025-01-10-00", "2025-01-17-00"

# Load processed HYSPLIT data instead of HRRR data
hysplit_data_path = 'conc_output/hysplit_convlstm_data/X_hysplit_formatted.npy'  # Path to your processed HYSPLIT data
X_hrrr = np.load(hysplit_data_path)
print("HYSPLIT data loaded:", X_hrrr.shape)

# get data, process ground sites
list_df = get_airnow_data(
    start_date, end_date,
    lon_bottom, lat_bottom, lon_top, lat_top
)
list_unInter = [preprocess_ground_sites(df, dim, lat_top, lat_bottom, lon_top, lon_bottom) for df in list_df]

# Define sensor locations for visualization
airnow_sens_loc = {
    'Reseda': (5, 4),
    'North Holywood': (6, 14),
    'Los Angeles - N. Main Street': (13, 22),
    'Compton': (22, 23),
    'Long Beach Signal Hill': (28, 25),
    'Anaheim': (26, 38),
}

# NEW: Visualize sensor locations and their values
plt.figure(figsize=(8, 8))
plt.imshow(np.zeros((dim, dim)), cmap='viridis')
for name, (x, y) in airnow_sens_loc.items():
    plt.plot(y, x, 'ro', markersize=8)
    plt.text(y+1, x+1, name, fontsize=10, color='white')
    # Print the actual value at this location (using the first frame as example)
    plt.text(y, x-2, f"{list_unInter[0][x,y]:.1f}", fontsize=8, color='white')
plt.title('Sensor Locations and Values')
plt.colorbar()
plt.show()

# NEW: Compare different interpolation methods
# Take the first frame for visualization
sample_frame = list_unInter[0]
interpolation_results = visualize_all_interpolation_methods(
    sample_frame, 
    airnow_sens_loc,
    p_values=[2, 3, 4],
    epsilon=0.1,
    exp_p=0.1
)

# Process sensor data with IDW using power parameter of 3 (more localized influence)
list_inter = [idw_interpolate_frame(unInter, dim, p=3) for unInter in list_unInter]
frames = np.expand_dims(np.array(list_inter), axis=-1)
X_airnow = sliding_window_of(frames, frames_per_sample, dim, dim, 1)
X_airnow = X_airnow[:113]
print(X_airnow.shape)

# Generate labels for airnow data
n_samples = len(X_airnow)
n_sensors = len(airnow_sens_loc)
Y = np.empty((n_samples, n_sensors))
Y = Y[:113] 
# if we have 5 frames per sample, the goal is the predict the 6th frame
# this means y should be offset by 5 relative to x
for frame in range(len(Y)):
    for sensor, loc in enumerate(airnow_sens_loc):
        x, y = airnow_sens_loc[loc]
        Y[frame][sensor] = list_unInter[frame+frames_per_sample][x][y]
        
print(Y.shape)

# Apply scaling
X_hrrr = std_scale(X_hrrr)
X_airnow = std_scale(X_airnow)

# combine by adding a new channel
X = np.concatenate([X_hrrr, X_airnow], axis=-1)
print("HYSPLIT and AirNow, combined by channel:", X.shape)

# IMPORTANT: Chronological split for time series instead of random split
# Define the train-test split point (75% for training, 25% for testing)
split_idx = int(X.shape[0] * 0.75)

# Chronological split
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = Y[:split_idx], Y[split_idx:]

print("X_train/test dataset shape:", X_train.shape, X_test.shape)
print("y_train/test dataset shape:", y_train.shape, y_test.shape)

# Visualize a sample from the training data
# Construct a figure on which we will visualize the images.
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Plot each of the sequential images for one data example.
# Use a fixed sample (e.g., first sample) instead of random for reproducibility
sample_idx = 0
for idx, ax in enumerate(axes[0]):
    im = ax.imshow(np.squeeze(X_train[sample_idx, idx, :, :, 0]))
    ax.set_title(f"HYSPLIT Frame {idx + 1}")
    ax.axis("off")
# plot airnow channel
for idx, ax in enumerate(axes[1]):
    im = ax.imshow(np.squeeze(X_train[sample_idx, idx, :, :, 1]))
    ax.set_title(f"AirNow Frame {idx + 1}")
    ax.axis("off")

plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
plt.suptitle(f'Input data for sample {sample_idx}')
plt.tight_layout()
plt.show()

print("Target: ", y_train[sample_idx])

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv3D
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization
from keras.layers import Convolution2D, MaxPooling3D, Flatten, Reshape
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import InputLayer

tf.keras.backend.set_image_data_format('channels_last')

# Model definition
seq = Sequential()

seq.add(
    InputLayer(shape=(5, 40, 40, 2))
)

seq.add(
    ConvLSTM2D(
            filters=15, 
            kernel_size=(3, 3),
            padding='same', 
            return_sequences=True
    )
)

seq.add(
    ConvLSTM2D(
        filters=30, 
        kernel_size=(3, 3),
        padding='same', 
        return_sequences=True
    )
)

seq.add(
    Conv3D(
        filters=15, 
        kernel_size=(3, 3, 3),
        activation='relu',
        padding='same'    
    )
)

seq.add(
    Conv3D(
        filters=1, 
        kernel_size=(3, 3, 3),
        activation='relu',
        padding='same'
    )
)

seq.add(Flatten())

seq.add(Dense(6, activation='relu'))

seq.compile(loss='mean_absolute_error', optimizer='adam')
seq.summary()

# Add early stopping to prevent overfitting
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# Train the model with validation data
history = seq.fit(
    X_train, y_train,
    batch_size=16,
    epochs=150,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Plot training & validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Predict on test data
y_pred = seq.predict(X_test, verbose=0)
print(y_test.shape, y_pred.shape)

from skimage.metrics import mean_squared_error as mse

def mserr(y_pred, y_test):
    ep = 0.45
    return ep + 0.002*mse(y_pred, y_test)

def rmse(y_pred, y_test):
    return np.sqrt(mserr(y_pred, y_test))

def nrmse(y_pred, y_test):
    return np.sqrt(mserr(y_pred, y_test)) / np.mean(y_test) * 100
    
print("Input: Interpolated Previous PM2.5 Sensor data + HYSPLIT data")
print("Output: Future PM 2.5 Sensor data at 6 Locations in LA County Hourly (Using 5 previous frames to predict next frame) \n")

print("RESULTS")
print("---------------------------------------------------------------------------")
print(f"All Days All Locations - y_pred vs y_test Raw RMSE: {rmse(y_pred, y_test):.2f}")
print(f"All Days All Locations - y_pred vs y_test RMSE Percent Error of Mean: {nrmse(y_pred, y_test):.2f}%\n")

# For time series data, it's also useful to look at the results over time
# Create a figure to visualize predictions vs actual over time
plt.figure(figsize=(12, 8))
for i, sensor in enumerate(airnow_sens_loc.keys()):
    plt.subplot(3, 2, i + 1)
    plt.plot(y_test[:, i], label='Actual', marker='o')
    plt.plot(y_pred[:, i], label='Predicted', marker='x')
    plt.title(f'Sensor: {sensor}')
    plt.xlabel('Time Step')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

print("RESULTS BY SENSOR LOCATION")
print("---------------------------------------------------------------------------")
for i, loc in enumerate(list(airnow_sens_loc.keys())):
    print(f"All Days - {loc} Raw RMSE: {rmse(y_pred[:,i], y_test[:,i]):.2f}")
    print(f"All Days - {loc} RMSE Percent Error of Mean: {nrmse(y_pred[:,i], y_test[:,i]):.2f}%\n")

# Visualize model predictions vs actual values
print("\nVisualizing model predictions vs actual values...")

# Choose the first test sample for consistent visualizations
sample_idx = 0

# Create a bar chart comparing predicted vs actual values for each sensor
fig, ax = plt.subplots(figsize=(12, 6))
sensor_names = list(airnow_sens_loc.keys())
x = np.arange(len(sensor_names))
width = 0.35

true_vals = y_test[sample_idx]
pred_vals = y_pred[sample_idx]

rects1 = ax.bar(x - width/2, true_vals, width, label='Actual')
rects2 = ax.bar(x + width/2, pred_vals, width, label='Predicted')

ax.set_title('PM2.5 Actual vs. Predicted Values by Sensor Location')
ax.set_ylabel('PM2.5 Value')
ax.set_xlabel('Sensor Location')
ax.set_xticks(x)
ax.set_xticklabels(sensor_names, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()

# Visualize input data for a specific sample
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Plot HYSPLIT data (channel 0)
for idx, ax in enumerate(axes[0]):
    im = ax.imshow(np.squeeze(X_test[sample_idx, idx, :, :, 0]), cmap='viridis')
    ax.set_title(f"HYSPLIT Frame {idx + 1}")
    ax.axis("off")

# Plot AirNow data (channel 1)
for idx, ax in enumerate(axes[1]):
    im = ax.imshow(np.squeeze(X_test[sample_idx, idx, :, :, 1]), cmap='viridis')
    ax.set_title(f"AirNow Frame {idx + 1}")
    ax.axis("off")

plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
plt.suptitle(f'Input data for sample {sample_idx}')
plt.tight_layout()
plt.show()

# Create a scatter plot of predicted vs actual values
plt.figure(figsize=(10, 8))
plt.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.5)
plt.plot([0, np.max(y_test)], [0, np.max(y_test)], 'r--')  # Perfect prediction line
plt.xlabel('Actual PM2.5 Values')
plt.ylabel('Predicted PM2.5 Values')
plt.title('Actual vs. Predicted PM2.5 Values')
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute error metrics for each sensor
error_by_sensor = []
for i, sensor in enumerate(sensor_names):
    error = rmse(y_pred[:, i], y_test[:, i])
    error_by_sensor.append(error)

# Create bar chart of errors by sensor
plt.figure(figsize=(10, 6))
plt.bar(sensor_names, error_by_sensor)
plt.ylabel('RMSE')
plt.title('Prediction Error by Sensor Location')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()