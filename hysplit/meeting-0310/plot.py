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
import imageio
from PIL import Image, ImageDraw, ImageFont

def read_hysplit_file(filename):
    """
    Read a HYSPLIT cdump file and return a pandas DataFrame
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    pattern = r'DAY HR\s+LAT\s+LON TEST00100\n([\s\S]+?)(?:\n\s*\n|\Z)'
    matches = re.findall(pattern, content)
    
    if not matches:
        print(f"Warning: No data found in {filename}")
        return None
    
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
    match = re.search(r'cdump_(\d+)_(\d+)', filename)
    if match:
        file_day, file_hour = match.groups()
        df['file_day'] = int(file_day)
        df['file_hour'] = int(file_hour)
        df['time_label'] = f"Day {file_day}, Hour {file_hour}"
    
    return df

def create_animation(data, output_dir='plots', fps=2):
    """
    Create an animated GIF of HYSPLIT concentration data over time
    """
    if data.empty:
        print("No data to animate")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get overall data limits with padding
    min_lat, max_lat = data['lat'].min() - 0.5, data['lat'].max() + 0.5
    min_lon, max_lon = data['lon'].min() - 0.5, data['lon'].max() + 0.5
    
    # Calculate the overall concentration range for consistent coloring
    vmin = data['concentration'].min()
    vmax = data['concentration'].max()
    
    # First, sort time labels by day and hour
    day_hour_tuples = []
    for label in data['time_label'].unique():
        match = re.search(r'Day (\d+), Hour (\d+)', label)
        if match:
            day, hour = int(match.group(1)), int(match.group(2))
            day_hour_tuples.append((day, hour, label))
    
    sorted_labels = [label for _, _, label in sorted(day_hour_tuples)]
    
    # Create a temporary directory for frames
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_files = []
    
    # Generate a frame for each time period
    for i, time_label in enumerate(sorted_labels):
        print(f"Creating frame {i+1}/{len(sorted_labels)} - {time_label}")
        time_data = data[data['time_label'] == time_label]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), 
                               subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linestyle=':')
        ax.add_feature(cfeature.BORDERS)
        
        # Set map extent
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Create scatter plot
        scatter = ax.scatter(
            time_data['lon'], 
            time_data['lat'],
            c=time_data['concentration'],
            s=50,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            alpha=0.8,
            edgecolor='k',
            linewidth=0.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Concentration')
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle=':', color='gray')
        gl.top_labels = False
        gl.right_labels = False
        
        plt.title(f'HYSPLIT Concentration - {time_label}', fontsize=14)
        
        frame_file = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.tight_layout()
        plt.savefig(frame_file, dpi=150)
        plt.close()
        
        frame_files.append(frame_file)
    
    print("Creating GIF animation...")
    images = []
    for file in frame_files:
        img = imageio.imread(file)
        images.append(img)
    
    gif_path = os.path.join(output_dir, 'concentration_animation.gif')
    imageio.mimsave(gif_path, images, fps=fps, loop=0)
    
    frames_pil = [Image.open(f) for f in frame_files]
    
    enhanced_gif_path = os.path.join(output_dir, 'concentration_animation_hq.gif')
    frames_pil[0].save(
        enhanced_gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        optimize=False,
        duration=int(1000/fps),  # milliseconds per frame
        loop=0
    )
    
    print(f"Animation saved to {gif_path} and {enhanced_gif_path}")
    print(f"Frame rate: {fps} frames per second")
    
    return gif_path

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
    
    print(combined_data)
    # # Create the animation
    # print("Creating animation...")
    # gif_path = create_animation(combined_data, fps=2)
    
    # print(f"Animation complete! GIF saved to: {gif_path}")
    # print("You can open the HTML file in a browser for better viewing.")

if __name__ == "__main__":
    main()