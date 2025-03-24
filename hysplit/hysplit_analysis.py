import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def read_tdump(filepath):
    """
    Read a HYSPLIT trajectory dump file (tdump).
    
    Parameters:
    -----------
    filepath : str
        Path to the tdump file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing trajectory data
    """
    # First, parse the header to determine where data starts
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the line that starts the actual data
    data_start_line = 0
    for i, line in enumerate(lines):
        if line.strip() and line[0] != ' ' and not line.startswith('1 '):
            data_start_line = i + 1
            break
    
    # Extract header info if needed
    header_info = lines[:data_start_line]
    
    # Parse the trajectory data
    # The format is typically:
    # trajectory_number, meteorology_grid, year, month, day, hour, minute, 
    # forecast_hour, age_hour, lat, lon, altitude(m_AGL), pressure
    
    # Initialize lists to store data
    traj_data = []
    
    # Parse trajectory data lines
    i = data_start_line
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this is a trajectory header line (typically starts with a trajectory number)
        if line and line.split()[0].isdigit():
            traj_num = int(line.split()[0])
            grid_num = int(line.split()[1])
            
            # Parse trajectory points
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().split()[0].isdigit():
                fields = lines[i].strip().split()
                
                # Construct a datetime object
                year = int(fields[0])
                month = int(fields[1])
                day = int(fields[2])
                hour = int(fields[3])
                minute = int(fields[4])
                
                # Handle 2-digit years
                if year < 100:
                    year += 2000 if year < 50 else 1900
                    
                # Extract other fields
                forecast_hour = int(fields[5])
                age_hour = int(fields[6])
                lat = float(fields[7])
                lon = float(fields[8])
                height = float(fields[9])
                
                # Additional fields if present
                pressure = float(fields[10]) if len(fields) > 10 else np.nan
                
                # Convert to dict
                point_data = {
                    'trajectory': traj_num,
                    'grid': grid_num,
                    'datetime': datetime(year, month, day, hour, minute),
                    'forecast_hour': forecast_hour,
                    'age_hour': age_hour,
                    'lat': lat,
                    'lon': lon,
                    'height': height,
                    'pressure': pressure
                }
                traj_data.append(point_data)
                
                i += 1
        else:
            i += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(traj_data)
    
    return df

def read_cdump(filepath):
    """
    Read a HYSPLIT concentration dump file (cdump).
    
    Parameters:
    -----------
    filepath : str
        Path to the cdump file
        
    Returns:
    --------
    dict
        Dictionary containing concentration data and grid information
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header information
    grid_info = {}
    
    # First line contains grid info
    grid_header = lines[0].strip().split()
    n_source_grids = int(grid_header[0])
    n_pollutants = int(grid_header[1])
    
    # Skip to concentration grid info
    line_index = 1
    
    for _ in range(n_source_grids):
        for pollutant in range(n_pollutants):
            # Parse time period info
            time_header = lines[line_index].strip().split()
            line_index += 1
            
            # Parse grid definition
            grid_def = lines[line_index].strip().split()
            line_index += 1
            
            n_lat = int(grid_def[0])
            lat_spacing = float(grid_def[1])
            lat_start = float(grid_def[2])
            
            n_lon = int(grid_def[3])
            lon_spacing = float(grid_def[4])
            lon_start = float(grid_def[5])
            
            # There may be level information
            n_levels = int(grid_def[6]) if len(grid_def) > 6 else 1
            
            grid_info = {
                'n_lat': n_lat,
                'lat_spacing': lat_spacing,
                'lat_start': lat_start,
                'n_lon': n_lon,
                'lon_spacing': lon_spacing,
                'lon_start': lon_start,
                'n_levels': n_levels
            }
            
            # Create arrays for lat/lon coordinates
            lats = np.array([lat_start + i * lat_spacing for i in range(n_lat)])
            lons = np.array([lon_start + i * lon_spacing for i in range(n_lon)])
            
            # Read concentration data for this time period/pollutant
            conc_data = []
            
            for level in range(n_levels):
                level_data = np.zeros((n_lat, n_lon))
                
                for lat_idx in range(n_lat):
                    values_read = 0
                    row_data = []
                    
                    while values_read < n_lon:
                        data_line = lines[line_index].strip().split()
                        line_index += 1
                        
                        for val in data_line:
                            row_data.append(float(val))
                            values_read += 1
                            
                            if values_read >= n_lon:
                                break
                    
                    level_data[lat_idx, :] = row_data
                
                conc_data.append(level_data)
            
    # Create a dictionary with all the information
    result = {
        'grid_info': grid_info,
        'lats': lats,
        'lons': lons,
        'concentration': np.array(conc_data)
    }
    
    return result

def plot_trajectory(tdump_data, title="HYSPLIT Trajectory"):
    """
    Plot trajectory data from a tdump file.
    
    Parameters:
    -----------
    tdump_data : pandas.DataFrame
        DataFrame from read_tdump function
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the trajectory on a map
    unique_trajectories = tdump_data['trajectory'].unique()
    
    for traj in unique_trajectories:
        traj_data = tdump_data[tdump_data['trajectory'] == traj]
        
        # Plot lat/lon
        ax1.plot(traj_data['lon'], traj_data['lat'], '-o', linewidth=2, markersize=4)
        
        # Mark start point with a star
        start_point = traj_data.iloc[0]
        ax1.plot(start_point['lon'], start_point['lat'], '*', markersize=12, color='red')
        
        # Mark end point with a square
        end_point = traj_data.iloc[-1]
        ax1.plot(end_point['lon'], end_point['lat'], 's', markersize=8, color='blue')
    
    # Add grid lines
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and title
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_title(title)
    
    # Plot height profile
    for traj in unique_trajectories:
        traj_data = tdump_data[tdump_data['trajectory'] == traj]
        ax2.plot(traj_data['age_hour'], traj_data['height'], '-o', linewidth=2, markersize=4)
    
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Trajectory Age (hours)')
    ax2.set_ylabel('Height (m AGL)')
    ax2.set_title('Trajectory Height Profile')
    
    plt.tight_layout()
    return fig

def plot_concentration(cdump_data, level=0, log_scale=True, title="HYSPLIT Concentration"):
    """
    Plot concentration data from a cdump file.
    
    Parameters:
    -----------
    cdump_data : dict
        Dictionary from read_cdump function
    level : int
        Vertical level to plot
    log_scale : bool
        Whether to use log scale for concentration values
    title : str
        Plot title
    """
    lats = cdump_data['lats']
    lons = cdump_data['lons']
    conc = cdump_data['concentration'][level]
    
    # Create a meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use log scale if selected
    if log_scale:
        # Add a small value to avoid log(0)
        conc_plot = np.log10(conc + 1e-10)
        vmin = max(np.min(conc_plot[conc_plot > -10]), -10)  # Avoid -inf
        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(vmin, np.max(conc_plot))
    else:
        conc_plot = conc
        vmin = 0
        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(vmin, np.max(conc_plot))
    
    # Create contour plot
    contour = ax.contourf(lon_grid, lat_grid, conc_plot, cmap=cmap, norm=norm, levels=20)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    if log_scale:
        cbar.set_label('log10(Concentration)')
    else:
        cbar.set_label('Concentration')
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def analyze_tdump_cdump(tdump_path, cdump_path, output_dir="hysplit_analysis"):
    """
    Analyze both trajectory (tdump) and concentration (cdump) files and save plots.
    
    Parameters:
    -----------
    tdump_path : str
        Path to tdump file
    cdump_path : str
        Path to cdump file
    output_dir : str
        Directory to save output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read tdump file
    print(f"Reading trajectory file: {tdump_path}")
    try:
        traj_data = read_tdump(tdump_path)
        print(f"Found {len(traj_data['trajectory'].unique())} trajectories with {len(traj_data)} total points")
        
        # Plot trajectory
        fig_traj = plot_trajectory(traj_data, title="HYSPLIT Trajectory Analysis")
        fig_traj.savefig(os.path.join(output_dir, "trajectory_analysis.png"), dpi=300)
        print(f"Saved trajectory plot to {os.path.join(output_dir, 'trajectory_analysis.png')}")
        
        # Save trajectory data to CSV
        traj_csv_path = os.path.join(output_dir, "trajectory_data.csv")
        traj_data.to_csv(traj_csv_path, index=False)
        print(f"Saved trajectory data to {traj_csv_path}")
        
    except Exception as e:
        print(f"Error reading/processing tdump file: {e}")
    
    # Read cdump file
    print(f"\nReading concentration file: {cdump_path}")
    try:
        conc_data = read_cdump(cdump_path)
        grid_info = conc_data['grid_info']
        print(f"Concentration grid: {grid_info['n_lat']}x{grid_info['n_lon']} points, {grid_info['n_levels']} vertical levels")
        
        # Plot concentration
        fig_conc = plot_concentration(conc_data, level=0, title="HYSPLIT Concentration Analysis")
        fig_conc.savefig(os.path.join(output_dir, "concentration_analysis.png"), dpi=300)
        print(f"Saved concentration plot to {os.path.join(output_dir, 'concentration_analysis.png')}")
        
        # Save concentration data to a binary file
        conc_data_path = os.path.join(output_dir, "concentration_data.npz")
        np.savez(conc_data_path, 
                 lats=conc_data['lats'], 
                 lons=conc_data['lons'], 
                 concentration=conc_data['concentration'],
                 grid_info=np.array([grid_info[k] for k in sorted(grid_info.keys())]))
        print(f"Saved concentration data to {conc_data_path}")
        
    except Exception as e:
        print(f"Error reading/processing cdump file: {e}")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    # Example usage
    tdump_file = "tdump"
    cdump_file = "cdump"
    output_directory = "hysplit_analysis"
    
    analyze_tdump_cdump(tdump_file, cdump_file, output_directory)