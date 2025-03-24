import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import ScalarFormatter
import json
from datetime import datetime

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

def create_lambert_conformal_plot(hysplit_data, output_path='lambert_single.png'):
    """
    Create a plot of HYSPLIT data using Lambert Conformal projection
    
    Parameters:
    -----------
    hysplit_data : DataFrame or path to cdump file
        HYSPLIT concentration data with lat, lon, concentration columns
    output_path : str
        Path to save the output image
    """
    # Process input data if it's a file path
    if isinstance(hysplit_data, str):
        data = read_hysplit_file(hysplit_data)
    else:
        data = hysplit_data
    
    # Define the Lambert Conformal projection
    # These parameters work well for North America
    central_longitude = -97.0
    standard_parallels = (33.0, 45.0)
    
    lambert_proj = ccrs.LambertConformal(
        central_longitude=central_longitude,
        standard_parallels=standard_parallels
    )
    
    # Create figure with Lambert Conformal projection
    fig, ax = plt.subplots(
        figsize=(12, 9),
        subplot_kw={'projection': lambert_proj}
    )
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7)
    
    # Determine appropriate extent based on data
    # Add some padding around the data points
    buffer = 2.0  # degrees
    min_lat = data['lat'].min() - buffer
    max_lat = data['lat'].max() + buffer
    min_lon = data['lon'].min() - buffer
    max_lon = data['lon'].max() + buffer
    
    # Set map extent
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Apply log normalization to concentration values
    epsilon = np.finfo(float).eps
    vmin = data['concentration'].min() + epsilon
    vmax = data['concentration'].max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # Plot the concentration data
    scatter = ax.scatter(
        data['lon'], 
        data['lat'],
        c=data['concentration'],
        s=15,  # marker size
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        norm=norm,
        alpha=0.8,
        edgecolor='none'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label('Concentration (Log Scale)')
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle=':', color='gray')
    gl.top_labels = False
    gl.right_labels = False
    
    # Set title
    plt.title('HYSPLIT Concentration - Lambert Conformal Projection', fontsize=14)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Lambert Conformal projection plot saved to {output_path}")

def create_lambert_hourly_panels(data, output_dir='lambert_plots'):
    """
    Create a multi-panel plot of hourly HYSPLIT data using Lambert Conformal projection
    
    Parameters:
    -----------
    data : DataFrame
        Combined HYSPLIT data with time_step column
    output_dir : str
        Directory to save the output images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique time steps and sort them with hour 16 first
    time_steps = sorted(data['time_step'].unique(), key=lambda x: (int(x) < 16, int(x)))
    
    # Define the Lambert Conformal projection
    central_longitude = -97.0
    standard_parallels = (33.0, 45.0)
    
    lambert_proj = ccrs.LambertConformal(
        central_longitude=central_longitude,
        standard_parallels=standard_parallels
    )
    
    # Create single plots for each hour
    for hour in time_steps:
        print(f"Creating Lambert plot for hour {hour}...")
        hour_data = data[data['time_step'] == hour]
        
        # Create figure with Lambert Conformal projection
        fig, ax = plt.subplots(
            figsize=(10, 8),
            subplot_kw={'projection': lambert_proj}
        )
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.7)
        
        # Determine appropriate extent based on data
        buffer = 2.0  # degrees
        min_lat = hour_data['lat'].min() - buffer
        max_lat = hour_data['lat'].max() + buffer
        min_lon = hour_data['lon'].min() - buffer
        max_lon = hour_data['lon'].max() + buffer
        
        # Set map extent
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Apply log normalization to concentration values
        epsilon = np.finfo(float).eps
        vmin = data['concentration'].min() + epsilon  # Use global min for consistency
        vmax = data['concentration'].max()  # Use global max for consistency
        norm = LogNorm(vmin=vmin, vmax=vmax)
        
        # Plot the concentration data
        scatter = ax.scatter(
            hour_data['lon'], 
            hour_data['lat'],
            c=hour_data['concentration'],
            s=15,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            norm=norm,
            alpha=0.8,
            edgecolor='none'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label('Concentration (Log Scale)')
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle=':', color='gray')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set title
        plt.title(f'HYSPLIT Concentration - Hour {hour}', fontsize=14)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hour_{int(hour):02d}_lambert.png'), dpi=300)
        plt.close()
    
    # Create multi-panel plot
    # Determine layout
    n_hours = len(time_steps)
    cols = min(4, n_hours)
    rows = int(np.ceil(n_hours / cols))
    
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3.5, rows * 3),
        subplot_kw={'projection': lambert_proj}
    )
    
    # Make sure axes is a 2D array
    if rows == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Apply log normalization to concentration values
    epsilon = np.finfo(float).eps
    vmin = data['concentration'].min() + epsilon
    vmax = data['concentration'].max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # Process each hour
    for i, hour in enumerate(time_steps):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        hour_data = data[data['time_step'] == hour]
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.2)
        
        # Set map extent (use global extent for consistency)
        buffer = 2.0
        min_lat = data['lat'].min() - buffer
        max_lat = data['lat'].max() + buffer
        min_lon = data['lon'].min() - buffer
        max_lon = data['lon'].max() + buffer
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Plot data
        scatter = ax.scatter(
            hour_data['lon'], 
            hour_data['lat'],
            c=hour_data['concentration'],
            s=8,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            norm=norm,
            alpha=0.8,
            edgecolor='none'
        )
        
        # Add panel title
        ax.set_title(f'Hour {hour}', fontsize=10)
        
        # Add light gridlines
        gl = ax.gridlines(draw_labels=False, linewidth=0.2, linestyle=':', color='gray')
    
    # Hide empty subplots
    for i in range(n_hours, rows * cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col])
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Concentration (Log Scale)')
    
    # Add figure title
    fig.suptitle('HYSPLIT Concentration - Lambert Conformal Projection', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'all_hours_lambert.png'), dpi=300)
    plt.close()
    
    print(f"Lambert Conformal panels saved to {output_dir}")

def create_gif_animation(data, output_dir='lambert_plots', fps=2):
    """
    Create an animated GIF of the Lambert Conformal projection plots
    
    Parameters:
    -----------
    data : DataFrame
        Combined HYSPLIT data with time_step column
    output_dir : str
        Directory to save the output images and GIF
    fps : int
        Frames per second for the animation
    """
    try:
        import imageio
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Error: imageio and/or PIL packages not installed. Cannot create GIF.")
        print("Install with: pip install imageio pillow")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique time steps and sort them with hour 16 first
    time_steps = sorted(data['time_step'].unique(), key=lambda x: (int(x) < 16, int(x)))
    
    # Define the Lambert Conformal projection
    central_longitude = -97.0
    standard_parallels = (33.0, 45.0)
    
    lambert_proj = ccrs.LambertConformal(
        central_longitude=central_longitude,
        standard_parallels=standard_parallels
    )
    
    # Prepare for GIF creation
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Apply log normalization to concentration values
    epsilon = np.finfo(float).eps
    vmin = data['concentration'].min() + epsilon
    vmax = data['concentration'].max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # Get global extent for consistent frames
    buffer = 2.0
    min_lat = data['lat'].min() - buffer
    max_lat = data['lat'].max() + buffer
    min_lon = data['lon'].min() - buffer
    max_lon = data['lon'].max() + buffer
    
    frame_files = []
    
    # Create a frame for each time step
    for i, hour in enumerate(time_steps):
        print(f"Creating frame {i+1}/{len(time_steps)} - Hour {hour}")
        hour_data = data[data['time_step'] == hour]
        
        # Create figure with Lambert Conformal projection
        fig, ax = plt.subplots(
            figsize=(10, 8),
            subplot_kw={'projection': lambert_proj}
        )
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linestyle=':')
        ax.add_feature(cfeature.BORDERS)
        
        # Set map extent
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Plot the concentration data
        scatter = ax.scatter(
            hour_data['lon'], 
            hour_data['lat'],
            c=hour_data['concentration'],
            s=15,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            norm=norm,
            alpha=0.8,
            edgecolor='k',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Concentration')
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Add grid lines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle=':', color='gray')
        gl.top_labels = False
        gl.right_labels = False
        
        # Add title
        plt.title(f'HYSPLIT Concentration - Hour {hour}', fontsize=14)
        
        # Save frame
        frame_file = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.tight_layout()
        plt.savefig(frame_file, dpi=150)
        plt.close()
        
        frame_files.append(frame_file)
    
    # Create GIF animation
    print("Creating GIF animation...")
    images = []
    for file in frame_files:
        img = imageio.imread(file)
        images.append(img)
    
    # Save GIF
    gif_path = os.path.join(output_dir, 'lambert_animation.gif')
    imageio.mimsave(gif_path, images, fps=fps, loop=0)
    
    # Create a higher quality GIF with PIL
    frames_pil = [Image.open(f) for f in frame_files]
    
    enhanced_gif_path = os.path.join(output_dir, 'lambert_animation_hq.gif')
    frames_pil[0].save(
        enhanced_gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        optimize=False,
        duration=int(1000/fps),  # milliseconds per frame
        loop=0
    )
    
    print(f"Animation saved to {gif_path} and {enhanced_gif_path}")

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
    
    # Create output directory
    output_dir = 'lambert_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create single combined plot
    print("Creating combined Lambert Conformal plot...")
    create_lambert_conformal_plot(combined_data, os.path.join(output_dir, 'combined_lambert.png'))
    
    # Create hourly panels
    print("Creating hourly Lambert Conformal plots...")
    create_lambert_hourly_panels(combined_data, output_dir)
    
    # Create animated GIF
    print("Creating Lambert Conformal animation...")
    create_gif_animation(combined_data, output_dir, fps=1.5)
    
    print(f"All Lambert Conformal plots saved to {output_dir}")
    print("Done!")

def create_robinson_plot(hysplit_data, output_path='robinson_single.png'):
    """
    Create a plot of HYSPLIT data using Robinson projection
    
    Parameters:
    -----------
    hysplit_data : DataFrame or path to cdump file
        HYSPLIT concentration data with lat, lon, concentration columns
    output_path : str
        Path to save the output image
    """
    # Process input data if it's a file path
    if isinstance(hysplit_data, str):
        data = read_hysplit_file(hysplit_data)
    else:
        data = hysplit_data
    
    # Define the Robinson projection
    robinson_proj = ccrs.Robinson(central_longitude=0)
    
    # Create figure with Robinson projection
    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={'projection': robinson_proj}
    )
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7)
    
    # Option to set global extent or focus on data
    global_view = False  # Set to True for global view
    
    if global_view:
        # For global view
        ax.set_global()
    else:
        # For data-focused view
        # Add some padding around the data points
        buffer = 10.0  # degrees (larger buffer for global projections)
        min_lat = max(-85, data['lat'].min() - buffer)  # Robinson projection has limits
        max_lat = min(85, data['lat'].max() + buffer)   # near the poles
        min_lon = data['lon'].min() - buffer
        max_lon = data['lon'].max() + buffer
        
        # Set map extent
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Apply log normalization to concentration values
    epsilon = np.finfo(float).eps
    vmin = data['concentration'].min() + epsilon
    vmax = data['concentration'].max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # Plot the concentration data
    scatter = ax.scatter(
        data['lon'], 
        data['lat'],
        c=data['concentration'],
        s=15,  # marker size
        transform=ccrs.PlateCarree(),  # Input coordinates are in lat/lon
        cmap='viridis',
        norm=norm,
        alpha=0.8,
        edgecolor='none'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label('Concentration (Log Scale)')
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle=':', color='gray')
    gl.top_labels = False
    gl.right_labels = False
    
    # Set title
    plt.title('HYSPLIT Concentration - Robinson Projection', fontsize=14)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Robinson projection plot saved to {output_path}")

def create_robinson_hourly_panels(data, output_dir='robinson_plots'):
    """
    Create a multi-panel plot of hourly HYSPLIT data using Robinson projection
    
    Parameters:
    -----------
    data : DataFrame
        Combined HYSPLIT data with time_step column
    output_dir : str
        Directory to save the output images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique time steps and sort them with hour 16 first
    time_steps = sorted(data['time_step'].unique(), key=lambda x: (int(x) < 16, int(x)))
    
    # Define the Robinson projection
    robinson_proj = ccrs.Robinson(central_longitude=0)
    
    # Create single plots for each hour
    for hour in time_steps:
        print(f"Creating Robinson plot for hour {hour}...")
        hour_data = data[data['time_step'] == hour]
        
        # Create figure with Robinson projection
        fig, ax = plt.subplots(
            figsize=(10, 6),
            subplot_kw={'projection': robinson_proj}
        )
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.7)
        
        # Determine appropriate extent based on data
        global_view = False  # Set to True for global view
        
        if global_view:
            # For global view
            ax.set_global()
        else:
            # For data-focused view
            buffer = 10.0  # degrees
            min_lat = max(-85, hour_data['lat'].min() - buffer)
            max_lat = min(85, hour_data['lat'].max() + buffer)
            min_lon = hour_data['lon'].min() - buffer
            max_lon = hour_data['lon'].max() + buffer
            
            # Set map extent
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Apply log normalization to concentration values
        epsilon = np.finfo(float).eps
        vmin = data['concentration'].min() + epsilon  # Use global min for consistency
        vmax = data['concentration'].max()  # Use global max for consistency
        norm = LogNorm(vmin=vmin, vmax=vmax)
        
        # Plot the concentration data
        scatter = ax.scatter(
            hour_data['lon'], 
            hour_data['lat'],
            c=hour_data['concentration'],
            s=15,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            norm=norm,
            alpha=0.8,
            edgecolor='none'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label('Concentration (Log Scale)')
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle=':', color='gray')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set title
        plt.title(f'HYSPLIT Concentration - Hour {hour} (Robinson Projection)', fontsize=14)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hour_{int(hour):02d}_robinson.png'), dpi=300)
        plt.close()
    
    # Create multi-panel plot
    # Determine layout
    n_hours = len(time_steps)
    cols = min(4, n_hours)
    rows = int(np.ceil(n_hours / cols))
    
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3.5, rows * 2.5),
        subplot_kw={'projection': robinson_proj}
    )
    
    # Make sure axes is a 2D array
    if rows == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Apply log normalization to concentration values
    epsilon = np.finfo(float).eps
    vmin = data['concentration'].min() + epsilon
    vmax = data['concentration'].max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # Process each hour
    for i, hour in enumerate(time_steps):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        hour_data = data[data['time_step'] == hour]
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.2)
        
        # Set map extent (use global extent or data-focused for consistency)
        global_view = False  # Set to True for global view
        
        if global_view:
            # For global view
            ax.set_global()
        else:
            # For data-focused view - use global extent for consistency
            buffer = 10.0
            min_lat = max(-85, data['lat'].min() - buffer)
            max_lat = min(85, data['lat'].max() + buffer)
            min_lon = data['lon'].min() - buffer
            max_lon = data['lon'].max() + buffer
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Plot data
        scatter = ax.scatter(
            hour_data['lon'], 
            hour_data['lat'],
            c=hour_data['concentration'],
            s=8,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            norm=norm,
            alpha=0.8,
            edgecolor='none'
        )
        
        # Add panel title
        ax.set_title(f'Hour {hour}', fontsize=10)
        
        # Add light gridlines
        gl = ax.gridlines(draw_labels=False, linewidth=0.2, linestyle=':', color='gray')
    
    # Hide empty subplots
    for i in range(n_hours, rows * cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col])
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Concentration (Log Scale)')
    
    # Add figure title
    fig.suptitle('HYSPLIT Concentration - Robinson Projection', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'all_hours_robinson.png'), dpi=300)
    plt.close()
    
    print(f"Robinson projection panels saved to {output_dir}")

def create_robinson_gif_animation(data, output_dir='robinson_plots', fps=2):
    """
    Create an animated GIF of the Robinson projection plots
    
    Parameters:
    -----------
    data : DataFrame
        Combined HYSPLIT data with time_step column
    output_dir : str
        Directory to save the output images and GIF
    fps : int
        Frames per second for the animation
    """
    try:
        import imageio
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Error: imageio and/or PIL packages not installed. Cannot create GIF.")
        print("Install with: pip install imageio pillow")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique time steps and sort them
    time_steps = sorted(data['time_step'].unique(), key=lambda x: (int(x) < 16, int(x)))
    
    # Define the Robinson projection
    robinson_proj = ccrs.Robinson(central_longitude=0)
    
    # Prepare for GIF creation
    frames_dir = os.path.join(output_dir, 'frames_robinson')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Apply log normalization to concentration values
    epsilon = np.finfo(float).eps
    vmin = data['concentration'].min() + epsilon
    vmax = data['concentration'].max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # Get global extent for consistent frames
    global_view = False  # Set to True for global view
    
    if not global_view:
        buffer = 10.0
        min_lat = max(-85, data['lat'].min() - buffer)
        max_lat = min(85, data['lat'].max() + buffer)
        min_lon = data['lon'].min() - buffer
        max_lon = data['lon'].max() + buffer
    
    frame_files = []
    
    # Create a frame for each time step
    for i, hour in enumerate(time_steps):
        print(f"Creating Robinson frame {i+1}/{len(time_steps)} - Hour {hour}")
        hour_data = data[data['time_step'] == hour]
        
        # Create figure with Robinson projection
        fig, ax = plt.subplots(
            figsize=(10, 6),
            subplot_kw={'projection': robinson_proj}
        )
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linestyle=':')
        ax.add_feature(cfeature.BORDERS)
        
        # Set map extent
        if global_view:
            ax.set_global()
        else:
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Plot the concentration data
        scatter = ax.scatter(
            hour_data['lon'], 
            hour_data['lat'],
            c=hour_data['concentration'],
            s=15,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            norm=norm,
            alpha=0.8,
            edgecolor='k',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Concentration')
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Add grid lines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle=':', color='gray')
        gl.top_labels = False
        gl.right_labels = False
        
        # Add title
        plt.title(f'HYSPLIT Concentration - Hour {hour} (Robinson Projection)', fontsize=14)
        
        # Save frame
        frame_file = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.tight_layout()
        plt.savefig(frame_file, dpi=150)
        plt.close()
        
        frame_files.append(frame_file)
    
    # Create GIF animation
    print("Creating Robinson GIF animation...")
    images = []
    for file in frame_files:
        img = imageio.imread(file)
        images.append(img)
    
    # Save GIF
    gif_path = os.path.join(output_dir, 'robinson_animation.gif')
    imageio.mimsave(gif_path, images, fps=fps, loop=0)
    
    # Create a higher quality GIF with PIL
    frames_pil = [Image.open(f) for f in frame_files]
    
    enhanced_gif_path = os.path.join(output_dir, 'robinson_animation_hq.gif')
    frames_pil[0].save(
        enhanced_gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        optimize=False,
        duration=int(1000/fps),  # milliseconds per frame
        loop=0
    )
    
    print(f"Robinson animation saved to {gif_path} and {enhanced_gif_path}")

def update_main_function():
    """
    Updated main function to include Robinson projection options
    """
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
    
    # Create Lambert output directory
    lambert_dir = 'lambert_plots'
    os.makedirs(lambert_dir, exist_ok=True)
    
    # Create Robinson output directory
    robinson_dir = 'robinson_plots'
    os.makedirs(robinson_dir, exist_ok=True)
    
    # Create Lambert plots
    print("\n--- Creating Lambert Conformal projections ---")
    print("Creating combined Lambert Conformal plot...")
    create_lambert_conformal_plot(combined_data, os.path.join(lambert_dir, 'combined_lambert.png'))
    
    print("Creating hourly Lambert Conformal plots...")
    create_lambert_hourly_panels(combined_data, lambert_dir)
    
    print("Creating Lambert Conformal animation...")
    create_gif_animation(combined_data, lambert_dir, fps=1.5)
    
    # Create Robinson plots
    print("\n--- Creating Robinson projections ---")
    print("Creating combined Robinson plot...")
    create_robinson_plot(combined_data, os.path.join(robinson_dir, 'combined_robinson.png'))
    
    print("Creating hourly Robinson plots...")
    create_robinson_hourly_panels(combined_data, robinson_dir)
    
    print("Creating Robinson animation...")
    create_robinson_gif_animation(combined_data, robinson_dir, fps=1.5)
    
    print(f"\nAll Lambert Conformal plots saved to {lambert_dir}")
    print(f"All Robinson plots saved to {robinson_dir}")
    print("Done!")

if __name__ == "__main__":
    update_main_function()