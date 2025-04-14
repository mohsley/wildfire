import requests
import json
import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

class AirNowData:
    '''
    Gets the AirNow Data.
    Pipeline:
        - Uses AirNow API to download the data as a list of dataframes
        - Extracts the ground site data and converts it into a grid
        - Interpolates the grid using IDW
        - Converts grids into numpy and adds a channel axis
            - (frames, row, col, channel)
        - Creates samples from a sliding window of frames
            - (samples, frames, row, col, channel)

    Members:
        data: The complete processed AirNow data
        ground_site_grids: The uninterpolated, ground-site gridded stations
        target_stations: The station values each sample wants to predict
        air_sens_loc: A dictionary of air sensor locations:
            - Location : (x, y)
    '''
    def __init__(
        self,
        start_date,
        end_date,
        extent,
        airnow_api_key=None,
        save_dir='data/airnow.json',
        frames_per_sample=1,
        dim=40,
        idw_power=2
    ):
        self.air_sens_loc = {}
        self.start_date = start_date
        self.end_date = end_date
        self.extent = extent
        self.dim = dim
        self.frames_per_sample = frames_per_sample
        self.idw_power = idw_power

        # Create necessary directories
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        
        # Get AirNow data
        list_df = self._get_airnow_data(
            start_date, end_date, 
            extent, 
            save_dir,
            airnow_api_key
        )
        
        # Generate simple grid data if we have no valid AirNow data
        if not list_df:
            print("Warning: No valid AirNow data available. Proceeding with minimal data structure.")
            # Create a simple data structure with minimal values
            simple_grid = np.zeros((dim, dim))
            
            # Create some dummy stations for structure
            stations = [
                {'name': 'Station1', 'x': 10, 'y': 10},
                {'name': 'Station2', 'x': 20, 'y': 30},
                {'name': 'Station3', 'x': 15, 'y': 25},
                {'name': 'Station4', 'x': 30, 'y': 10},
                {'name': 'Station5', 'x': 25, 'y': 15},
                {'name': 'Station6', 'x': 5, 'y': 35}
            ]
            
            # Add minimal values at station locations
            for station in stations:
                x, y = station['x'], station['y']
                simple_grid[x, y] = 10.0  # Default value
                self.air_sens_loc[station['name']] = (x, y)
            
            # Create a list with this grid repeated for each time step
            timestamps = pd.date_range(
                start=pd.to_datetime(start_date), 
                end=pd.to_datetime(end_date), 
                freq='1H'
            )
            
            ground_site_grids = [simple_grid.copy() for _ in range(len(timestamps))]
            interpolated_grids = ground_site_grids
        else:
            # Process normal AirNow data
            ground_site_grids = [
                self._preprocess_ground_sites(df, dim, extent) for df in list_df
            ]
            interpolated_grids = [
                self._interpolate_frame(frame, dim, self.idw_power) for frame in ground_site_grids
            ]
        
        # Continue with the rest of the pipeline
        frames = np.expand_dims(np.array(interpolated_grids), axis=-1)
        processed_ds = self._sliding_window_of(frames, frames_per_sample)

        self.data = processed_ds
        self.ground_site_grids = ground_site_grids
        
        # Get target stations or create default targets
        if self.air_sens_loc:
            self.target_stations = self._get_target_stations(
                self.data, self.ground_site_grids, self.air_sens_loc
            )
        else:
            # Create default target values if no stations were found
            n_samples = processed_ds.shape[0]
            n_stations = len(self.air_sens_loc)
            self.target_stations = np.zeros((n_samples, n_stations))

    def _get_airnow_data(
        self, 
        start_date, end_date, 
        extent, 
        save_dir, 
        airnow_api_key
    ):
        """
        Grabs the AirNow data from the API or loads from existing file.
        Returns a list of dataframes grouped by time, or an empty list if data is invalid.
        """
        lon_bottom, lon_top, lat_bottom, lat_top = extent
        
        # Get airnow data from the EPA
        if os.path.exists(save_dir):
            print(f"'{save_dir}' already exists; skipping request...")
        else:
            # Preprocess parameters
            date_start = pd.to_datetime(start_date).isoformat()[:13]
            date_end = pd.to_datetime(end_date).isoformat()[:13]
            bbox = f'{lon_bottom},{lat_bottom},{lon_top},{lat_top}'
            URL = "https://www.airnowapi.org/aq/data"

            # Parameters for the API
            PARAMS = {
                'startDate': date_start,
                'endDate': date_end,
                'parameters': 'PM25',
                'BBOX': bbox,
                'dataType': 'B',
                'format': 'application/json',
                'verbose': '1',
                'monitorType': '2',
                'includerawconcentrations': '1',
                'API_KEY': airnow_api_key
            }

            # Send request and save response
            try:
                response = requests.get(url=URL, params=PARAMS)
                airnow_data = response.json()
                with open(save_dir, 'w') as file:
                    json.dump(airnow_data, file)
                    print(f"JSON data saved to '{save_dir}'")
            except Exception as e:
                print(f"Error retrieving AirNow data: {e}")
                return []

        # Load and process the data
        try:
            with open(save_dir, 'r') as file:
                airnow_data = json.load(file)
            
            # Check if data has error message
            if isinstance(airnow_data, list) and len(airnow_data) > 0 and isinstance(airnow_data[0], dict):
                if 'WebServiceError' in airnow_data[0]:
                    print(f"Error from AirNow API: {airnow_data[0]['WebServiceError']}")
                    return []
            
            # Continue with normal processing if data looks valid
            airnow_df = pd.json_normalize(airnow_data)
            
            # Check for UTC column
            if 'UTC' not in airnow_df.columns:
                print("Error: 'UTC' column not found in AirNow data.")
                
                # Try to construct UTC from other fields if possible
                if 'DateObserved' in airnow_df.columns and 'HourObserved' in airnow_df.columns:
                    print("Attempting to construct UTC from DateObserved and HourObserved...")
                    try:
                        airnow_df['UTC'] = airnow_df.apply(
                            lambda row: pd.Timestamp(row['DateObserved']).replace(
                                hour=int(row['HourObserved'])
                            ).strftime('%Y-%m-%dT%H:%M'),
                            axis=1
                        )
                    except Exception as e:
                        print(f"Failed to construct UTC: {e}")
                        return []
                else:
                    print("Required columns for UTC construction not found.")
                    return []
            
            # Group by UTC
            try:
                list_df = [group for name, group in airnow_df.groupby('UTC')]
                return list_df
            except Exception as e:
                print(f"Error grouping by UTC: {e}")
                return []
                
        except Exception as e:
            print(f"Error processing AirNow data: {e}")
            return []

    def _preprocess_ground_sites(self, df, dim, extent):
        """
        Preprocess ground sites data into a grid.
        """
        lonMin, lonMax, latMin, latMax = extent
        latDist, lonDist = abs(latMax - latMin), abs(lonMax - lonMin)
        unInter = np.zeros((dim,dim))
        
        # Check if the required columns exist
        required_columns = ['Latitude', 'Longitude', 'Value', 'SiteName']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Missing required columns in dataframe. Available columns: {df.columns}")
            return unInter
            
        dfArr = np.array(df[required_columns])
        
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
                # save sensor site name and location
                sitename = dfArr[i,3]
                self.air_sens_loc[sitename] = (x, y)
        return unInter

    def _interpolate_frame(self, frame, dim, power=2):
        """
        Interpolates a frame using Inverse Distance Weighting (IDW) interpolation.
        
        Parameters:
        -----------
        frame : numpy array
            The input frame with non-zero values at station locations
        dim : int
            Grid dimension
        power : float
            Power parameter that controls how quickly the influence of a point decreases with distance
            
        Returns:
        --------
        interpolated : numpy array
            The interpolated grid
        """
        # Find non-zero values (station locations)
        x_list = []
        y_list = []
        values = []
        
        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                if frame[x, y] != 0:
                    x_list.append(x)
                    y_list.append(y)
                    values.append(frame[x, y])
        
        # If no sample points found, return the original frame
        if len(values) == 0:
            return frame
        
        # Convert to numpy arrays
        values = np.array(values)
        
        # Initialize the output grid
        Z = np.zeros((dim, dim))
        
        # Simple IDW implementation - loop through each grid cell
        for i in range(dim):
            for j in range(dim):
                # If this is a station point, use its value directly
                if frame[i, j] != 0:
                    Z[i, j] = frame[i, j]
                    continue
                
                # Calculate distances to all stations
                distances = np.array([np.sqrt((i - x)**2 + (j - y)**2) for x, y in zip(x_list, y_list)])
                
                # Handle potential division by zero
                if np.any(distances == 0):
                    # If we're exactly at a station point, use that value
                    Z[i, j] = values[np.where(distances == 0)[0][0]]
                else:
                    # Apply IDW formula: sum(value_i / distance_i^p) / sum(1 / distance_i^p)
                    weights = 1.0 / (distances ** power)
                    Z[i, j] = np.sum(weights * values) / np.sum(weights)
        
        return Z

    def _sliding_window_of(self, frames, frames_per_sample):
        """
        Uses a sliding window to bundle frames into samples.
        """
        n_frames, row, col, channels = frames.shape
        n_samples = max(1, n_frames - frames_per_sample + 1)
        samples = np.empty((n_samples, frames_per_sample, row, col, channels))
        
        for i in range(n_samples):
            end_idx = min(i + frames_per_sample, n_frames)
            if end_idx - i < frames_per_sample:
                # Not enough frames, repeat the last frame
                sample_frames = []
                for j in range(i, end_idx):
                    sample_frames.append(frames[j])
                # Fill the rest with the last frame
                for j in range(end_idx - i, frames_per_sample):
                    sample_frames.append(frames[end_idx-1] if end_idx > 0 else frames[0])
                samples[i] = np.array(sample_frames)
            else:
                samples[i] = np.array([frames[j] for j in range(i, i + frames_per_sample)])
            
        return samples

    def _get_target_stations(self, X, gridded_data, sensor_locations):
        """
        Gets the desired target stations to predict for a given list of samples.
        """
        n_samples, frames_per_sample = X.shape[0], X.shape[1] 
        n_sensors = len(sensor_locations)
        
        if n_sensors == 0:
            # Return empty array if no sensors
            return np.empty((n_samples, 0))
            
        Y = np.empty((n_samples, n_sensors))
        
        for sample in range(len(Y)):
            for sensor, loc in enumerate(sensor_locations):
                x, y = sensor_locations[loc]
                offset = sample + frames_per_sample
                
                if offset < len(gridded_data):
                    Y[sample][sensor] = gridded_data[offset][x][y]
                else:
                    # Use the last available data for out-of-bounds targets
                    Y[sample][sensor] = gridded_data[-1][x][y]

        return Y