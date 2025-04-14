#!/usr/bin/env python3
"""
Test script for airnowdata.py module
This script tests the AirNowData class to diagnose issues
"""

import os
import json
import pandas as pd
import numpy as np
import sys
import traceback

# Adjust this if the module is in a different location
sys.path.append('/mnt/d/school/wildfire/hysplit/organized/libs')

try:
    from airnowdata_module import AirNowData
except ImportError:
    print("Failed to import AirNowData class. Check if the module is in the correct location.")
    sys.exit(1)

def inspect_json_file(file_path):
    """Inspect the content of the JSON file"""
    print(f"\n=== Inspecting JSON file: {file_path} ===")
    
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: File does not exist: {file_path}")
            return
            
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        print(f"JSON file size: {os.path.getsize(file_path)} bytes")
        print(f"Number of records: {len(data)}")
        
        if len(data) > 0:
            print("\nSample record:")
            print(json.dumps(data[0], indent=2))
            
            # Check for UTC field
            if 'UTC' in data[0]:
                print("\nUTC field found in records.")
            else:
                print("\nWARNING: UTC field NOT found in records.")
                print("Available fields:")
                for key in data[0].keys():
                    print(f"- {key}")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        traceback.print_exc()

def test_direct_json_loading(file_path):
    """Try loading the JSON and grouping by UTC directly"""
    print(f"\n=== Testing direct JSON loading and grouping: {file_path} ===")
    
    try:
        with open(file_path, 'r') as file:
            airnow_data = json.load(file)
            
        airnow_df = pd.json_normalize(airnow_data)
        print("\nDataFrame columns:")
        for col in airnow_df.columns:
            print(f"- {col}")
        
        print(f"\nDataFrame shape: {airnow_df.shape}")
        
        if 'UTC' in airnow_df.columns:
            try:
                # Try grouping by UTC
                grouped = airnow_df.groupby('UTC')
                list_df = [group for name, group in grouped]
                print(f"Successfully grouped by UTC: {len(list_df)} groups")
                
                # Show sample of each group
                for i, df in enumerate(list_df[:3]):  # Show first 3 groups
                    print(f"\nGroup {i}, Shape: {df.shape}")
                    print(df.head(3))
            except Exception as e:
                print(f"Error grouping by UTC: {e}")
                traceback.print_exc()
        else:
            print("ERROR: UTC column not found in dataframe")
            
            # Check if we can construct UTC
            if 'DateObserved' in airnow_df.columns and 'HourObserved' in airnow_df.columns:
                print("\nTrying to construct UTC from DateObserved and HourObserved")
                try:
                    airnow_df['UTC'] = airnow_df.apply(
                        lambda row: pd.Timestamp(row['DateObserved']).replace(
                            hour=int(row['HourObserved'])
                        ).strftime('%Y-%m-%dT%H:%M'),
                        axis=1
                    )
                    print("UTC constructed successfully")
                    print(airnow_df['UTC'].head())
                except Exception as e:
                    print(f"Error constructing UTC: {e}")
                    traceback.print_exc()
    except Exception as e:
        print(f"Error in direct JSON processing: {e}")
        traceback.print_exc()

def test_airnow_data_class():
    """Test the AirNowData class"""
    print("\n=== Testing AirNowData class ===")
    
    # Set test parameters
    start_date = "2025-01-10-00"
    end_date = "2025-01-17-00"
    extent = (-118.6, -117.9, 33.6, 34.3)  # lon_min, lon_max, lat_min, lat_max
    save_dir = 'data/airnow.json'
    frames_per_sample = 5
    dim = 40
    
    # First check if the JSON file exists
    if os.path.exists(save_dir):
        print(f"JSON file exists at {save_dir}")
        inspect_json_file(save_dir)
        test_direct_json_loading(save_dir)
    else:
        print(f"WARNING: JSON file does not exist at {save_dir}")
    
    # Create a directory for the JSON file if it doesn't exist
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    # Debug the AirNowData initialization
    try:
        print("\nInitializing AirNowData class...")
        ad = AirNowData(
            start_date=start_date,
            end_date=end_date,
            extent=extent,
            airnow_api_key=None,
            save_dir=save_dir,
            frames_per_sample=frames_per_sample,
            dim=dim
        )
        
        print(f"AirNowData initialization successful")
        print(f"Data shape: {ad.data.shape}")
        print(f"Target stations shape: {ad.target_stations.shape}")
        print(f"Number of air sensor locations: {len(ad.air_sens_loc)}")
        
        # Check air_sens_loc dictionary
        print("\nAir sensor locations:")
        for name, loc in ad.air_sens_loc.items():
            print(f"- {name}: {loc}")
        
    except Exception as e:
        print(f"Error initializing AirNowData: {e}")
        traceback.print_exc()
        
        # Try to identify the specific issue
        try:
            # Check the __get_airnow_data method
            print("\nTesting __get_airnow_data method directly...")
            
            # Create a minimal instance
            minimal_ad = type('obj', (object,), {
                'air_sens_loc': {},
                '__get_airnow_data': AirNowData.__get_airnow_data
            })()
            
            list_df = minimal_ad.__get_airnow_data(
                start_date, end_date, extent, save_dir, None
            )
            
            print(f"Successfully got {len(list_df)} dataframes from __get_airnow_data")
            
        except Exception as e:
            print(f"Error in __get_airnow_data: {e}")
            traceback.print_exc()
            
            # Try to identify error in groupby
            try:
                print("\nTesting JSON load and groupby directly...")
                with open(save_dir, 'r') as file:
                    airnow_data = json.load(file)
                airnow_df = pd.json_normalize(airnow_data)
                print(f"DataFrame columns: {airnow_df.columns}")
                
                if 'UTC' not in airnow_df.columns:
                    print("ERROR: 'UTC' column missing from dataframe.")
                    print("Available columns:")
                    for col in airnow_df.columns:
                        print(f"- {col}")
                else:
                    list_df = [group for name, group in airnow_df.groupby('UTC')]
                    print(f"Groupby successful: {len(list_df)} groups")
            except Exception as inner_e:
                print(f"Error in JSON processing: {inner_e}")
                traceback.print_exc()
                

if __name__ == "__main__":
    print("=== AirNowData Module Tester ===")
    test_airnow_data_class()