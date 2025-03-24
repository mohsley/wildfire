import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.preprocessing import MinMaxScaler
import h5py
import tensorflow as tf
from datetime import datetime
import json

def read_hysplit_file(filename):
    """
    Read a HYSPLIT cdump file and return a pandas DataFrame
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract data after the header line
    pattern = r'DAY HR\s+LAT\s+LON TEST00100\n([\s\S]+?)(?:\n\s*\n|\Z)'
    matches = re.findall(pattern, content)
    
    if not matches:
        print(f"Warning: No data found in {filename}")
        return None
    
    # Process all data chunks
    all_data = []
    for data_chunk in matches:
        lines = data_chunk.strip().split('\n')
        chunk_data = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    day = int(parts[0])
                    hour = int(parts[1])
                    lat = float(parts[2])
                    lon = float(parts[3])
                    # Handle scientific notation (e.g., 0.78E-14)
                    concentration = float(parts[4].replace('E', 'e'))
                    
                    chunk_data.append({
                        'day': day,
                        'hour': hour,
                        'lat': lat,
                        'lon': lon,
                        'concentration': concentration,
                        'filename': os.path.basename(filename)
                    })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line in {filename}: {line}")
                    print(f"Error: {e}")
        
        all_data.extend(chunk_data)
    
    if not all_data:
        print(f"Warning: No valid data found in {filename}")
        return None
    
    df = pd.DataFrame(all_data)
    # Extract time information from filename (e.g., cdump_008_16)
    match = re.search(r'cdump_(\d+)_(\d+)', filename)
    if match:
        file_day, file_hour = match.groups()
        df['file_day'] = int(file_day)
        df['file_hour'] = int(file_hour)
        df['time_step'] = int(file_hour)  # Using hour as time step identifier
    
    return df

def prepare_convlstm_data(data, grid_size=64, output_dir='convlstm_data'):
    """
    Prepare HYSPLIT concentration data for ConvLSTM model input
    
    Args:
        data: DataFrame with HYSPLIT concentration data
        grid_size: Size of the grid (grid_size x grid_size)
        output_dir: Directory to save processed data
    
    Returns:
        Dictionary with processed data ready for ConvLSTM
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get spatial bounds
    min_lat, max_lat = data['lat'].min(), data['lat'].max()
    min_lon, max_lon = data['lon'].min(), data['lon'].max()
    
    # Add small padding
    lat_padding = (max_lat - min_lat) * 0.05
    lon_padding = (max_lon - min_lon) * 0.05
    min_lat -= lat_padding
    max_lat += lat_padding
    min_lon -= lon_padding
    max_lon += lon_padding
    
    # Get unique time steps
    time_steps = sorted(data['time_step'].unique())
    num_time_steps = len(time_steps)
    
    print(f"Processing {num_time_steps} time steps to {grid_size}x{grid_size} grids")
    
    # Create empty 4D array [time_steps, height, width, channels]
    # Channel 1: Concentration
    sequence_data = np.zeros((num_time_steps, grid_size, grid_size, 1))
    
    # Create a map from hour to sequence index
    hour_to_idx = {hour: idx for idx, hour in enumerate(time_steps)}
    
    # Process each time step
    for time_step in time_steps:
        time_data = data[data['time_step'] == time_step]
        
        # Create grid for this time step
        grid = np.zeros((grid_size, grid_size))
        
        # Create lat/lon bins
        lat_bins = np.linspace(min_lat, max_lat, grid_size + 1)
        lon_bins = np.linspace(min_lon, max_lon, grid_size + 1)
        
        # Digitize the data points into grid cells
        lat_indices = np.digitize(time_data['lat'], lat_bins) - 1
        lon_indices = np.digitize(time_data['lon'], lon_bins) - 1
        
        # Clip indices to valid range (0 to grid_size-1)
        lat_indices = np.clip(lat_indices, 0, grid_size - 1)
        lon_indices = np.clip(lon_indices, 0, grid_size - 1)
        
        # Map concentration values to grid cells (using maximum for overlapping points)
        for i, (lat_idx, lon_idx, conc) in enumerate(zip(
                lat_indices, lon_indices, time_data['concentration'])):
            # Take maximum value for any overlapping points
            grid[lat_idx, lon_idx] = max(grid[lat_idx, lon_idx], conc)
        
        # Store grid in sequence_data
        sequence_idx = hour_to_idx[time_step]
        sequence_data[sequence_idx, :, :, 0] = grid
    
    # Apply log transformation to handle wide range of concentration values
    # Add small epsilon to avoid log(0)
    epsilon = np.finfo(float).eps
    sequence_data_log = np.log1p(sequence_data + epsilon)
    
    # Normalize data to [0, 1] range
    scaler = MinMaxScaler()
    sequence_data_flat = sequence_data_log.reshape(-1, 1)
    sequence_data_norm_flat = scaler.fit_transform(sequence_data_flat)
    sequence_data_norm = sequence_data_norm_flat.reshape(sequence_data_log.shape)
    
    # Save normalization parameters for later use
    np.save(os.path.join(output_dir, 'scaler_min_max.npy'), 
            np.array([scaler.data_min_[0], scaler.data_max_[0]]))
    
    # Save spatial bounds for reference
    spatial_bounds = {
        'min_lat': min_lat,
        'max_lat': max_lat,
        'min_lon': min_lon,
        'max_lon': max_lon,
        'lat_bins': lat_bins,
        'lon_bins': lon_bins
    }
    np.save(os.path.join(output_dir, 'spatial_bounds.npy'), spatial_bounds)
    
    # Save time information
    time_info = {
        'time_steps': time_steps,
        'hour_to_idx': hour_to_idx
    }
    np.save(os.path.join(output_dir, 'time_info.npy'), time_info)
    
    # Save as NumPy array
    np.save(os.path.join(output_dir, 'sequence_data.npy'), sequence_data_norm)
    
    # Save as HDF5 file (alternative format often used for large datasets)
    h5f = h5py.File(os.path.join(output_dir, 'sequence_data.h5'), 'w')
    h5f.create_dataset('data', data=sequence_data_norm)
    h5f.create_dataset('data_original', data=sequence_data)
    h5f.close()
    
    # Create TensorFlow format
    tf_dataset = tf.data.Dataset.from_tensor_slices(sequence_data_norm)
    tf.data.experimental.save(tf_dataset, os.path.join(output_dir, 'tf_dataset'))
    
    # Save metadata
    metadata = {
        'grid_size': grid_size,
        'num_time_steps': num_time_steps,
        'min_lat': min_lat,
        'max_lat': max_lat,
        'min_lon': min_lon,
        'max_lon': max_lon,
        'min_concentration': data['concentration'].min(),
        'max_concentration': data['concentration'].max(),
        'log_transform_applied': True,
        'normalization_applied': True,
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save metadata as JSON
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Create a visualization of the processed data to verify
    visualize_processed_data(sequence_data_norm, time_steps, output_dir)
    
    print(f"Processed data saved to {output_dir}")
    print(f"Data shape: {sequence_data_norm.shape}")
    print(f"Files created: {os.listdir(output_dir)}")
    
    return {
        'data': sequence_data_norm,
        'metadata': metadata,
        'spatial_bounds': spatial_bounds,
        'time_info': time_info
    }

def visualize_processed_data(sequence_data, time_steps, output_dir):
    """Create visualizations of the processed data in original time order"""
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get the sequence of time steps in their original order
    # This will be like [16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, ..., 15]
    time_steps_sorted = sorted(time_steps, key=lambda x: (int(x) < 16, int(x)))
    
    # Create a mapping from time step to sequence index for reference
    time_step_to_idx = {time_step: idx for idx, time_step in enumerate(time_steps_sorted)}
    
    # Save the time step mapping for reference
    with open(os.path.join(output_dir, 'time_step_mapping.json'), 'w') as f:
        mapping_info = {
            "original_time_steps": [int(t) for t in time_steps_sorted],
            "sequence_indices": list(range(len(time_steps_sorted))),
            "mapping": {int(ts): idx for ts, idx in time_step_to_idx.items()}
        }
        json.dump(mapping_info, f, indent=4)
    
    # Visualize each time step
    for seq_idx, time_step in enumerate(time_steps_sorted):
        # Find the original index in the sequence_data array
        orig_idx = list(time_steps).index(time_step)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(sequence_data[orig_idx, :, :, 0], cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Normalized Concentration')
        plt.title(f'Hour {time_step}')
        plt.tight_layout()
        
        # Save image with original hour in filename
        plt.savefig(os.path.join(vis_dir, f'hour_{int(time_step):02d}.png'), dpi=150)
        plt.close()
    
    # Create a sequence of small multiples (4 columns)
    rows = int(np.ceil(len(time_steps) / 4))
    cols = min(4, len(time_steps))
    
    plt.figure(figsize=(cols * 3, rows * 3))
    
    # Plot in the original hour sequence
    for seq_idx, time_step in enumerate(time_steps_sorted):
        # Find original index in sequence_data
        orig_idx = list(time_steps).index(time_step)
        
        plt.subplot(rows, cols, seq_idx + 1)
        plt.imshow(sequence_data[orig_idx, :, :, 0], cmap='viridis', interpolation='nearest')
        plt.title(f'Hr {time_step}', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'all_hours.png'), dpi=200)
    plt.close()
    
    # Create an informative readme file
    with open(os.path.join(vis_dir, 'README.txt'), 'w') as f:
        f.write("HYSPLIT Concentration Data Visualization\n")
        f.write("======================================\n\n")
        f.write("The images in this directory show the normalized concentration data\n")
        f.write("prepared for ConvLSTM modeling.\n\n")
        f.write("Time progression follows the original hours from 16 through 15 (next day).\n\n")
        f.write("Sequence of hours:\n")
        for i, ts in enumerate(time_steps_sorted):
            f.write(f"  Sequence position {i}: Hour {ts}\n")
        f.write("\nThe data has been log-transformed and normalized to [0,1] range.\n")
        
def create_model_template(grid_size=64, num_time_steps=24):
    """Create a template ConvLSTM model structure for reference"""
    
    try:
        # This is a template only - you'll need to adjust parameters based on your specific needs
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
        
        # Define model
        model = Sequential()
        
        # ConvLSTM layers
        model.add(ConvLSTM2D(
            filters=64, 
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu',
            input_shape=(num_time_steps, grid_size, grid_size, 1)
        ))
        model.add(BatchNormalization())
        
        model.add(ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=False,
            activation='relu'
        ))
        model.add(BatchNormalization())
        
        # Output layer
        model.add(Conv2D(
            filters=1,
            kernel_size=(3, 3),
            activation='sigmoid',
            padding='same'
        ))
        
        # Compile model (for prediction example)
        model.compile(
            optimizer='adam',
            loss='mse'
        )
        
        # Save model summary to file
        model.summary()
        return model
    except Exception as e:
        print(f"Error creating model template: {e}")
        print("This is just a template and doesn't affect the data preparation.")
        return None

def main():
    # Find all cdump files
    cdump_files = glob.glob('cdump_*_*')
    
    if not cdump_files:
        print("No cdump files found in the current directory")
        print("Please make sure the files are in the format cdump_XXX_YY")
        return
    
    print(f"Found {len(cdump_files)} cdump files")
    
    # Read all files
    all_data = []
    for file in cdump_files:
        print(f"Reading {file}...")
        df = read_hysplit_file(file)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("No valid data found in any files")
        return
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Prepare data for ConvLSTM model
    grid_size = 64  # Adjust grid size as needed (64x64 is common)
    processed_data = prepare_convlstm_data(combined_data, grid_size=grid_size)
    
    # Create model template
    model = create_model_template(grid_size=grid_size, num_time_steps=len(processed_data['time_info']['time_steps']))
    
    # Save model template if created successfully
    if model is not None:
        try:
            model.save('convlstm_data/model_template.h5')
        except Exception as e:
            print(f"Error saving model template: {e}")
    
    print("\nData preparation complete!")
    print("\nNext steps for ConvLSTM training:")
    print("1. Load the prepared data from convlstm_data/sequence_data.npy or .h5")
    print("2. Decide on sequence length for input/output (e.g., use first 12 hours to predict next 12)")
    print("3. Split data into training/validation sets")
    print("4. Adjust the model template as needed")
    print("5. Train the model with your prepared sequences")

if __name__ == "__main__":
    main()