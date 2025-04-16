#!/usr/bin/env python3
"""
AirNowData Test Script

This script tests the AirNowData class, with special focus on diagnosing coordinate
conversion issues and ensuring sensors appear within the mask boundaries.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import AirNowData - adjust path if needed
sys.path.append('.')  # Add current directory to path
from airnowdata_module import AirNowData  # Rename this import to match your file

# Load environment variables for API key
load_dotenv()

def test_coordinate_conversion(extent, dim=200):
    """
    Test the coordinate conversion logic used in AirNowData to ensure points 
    are properly mapped to the grid.
    """
    print("\n=== Testing Coordinate Conversion ===")
    
    # Unpack extent
    lon_bottom, lon_top, lat_bottom, lat_top = extent
    
    # Calculate distance spans
    lat_dist = abs(lat_top - lat_bottom)
    lon_dist = abs(lon_top - lon_bottom)
    
    # Generate test points - corners, center, and arbitrary points
    test_points = [
        # Name, Lat, Lon, Expected to be in mask
        ("Bottom-Left", lat_bottom, lon_bottom, False),
        ("Bottom-Right", lat_bottom, lon_top, False),
        ("Top-Left", lat_top, lon_bottom, False),
        ("Top-Right", lat_top, lon_top, False),
        ("Center", (lat_bottom + lat_top)/2, (lon_bottom + lon_top)/2, True),
        # Add more test points as needed
    ]
    
    # Load mask if available
    mask_path = os.path.join('inputs', 'mask.npy')
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        mask_available = True
    else:
        print(f"Warning: Mask file not found at {mask_path}")
        mask_available = False
        mask = np.ones((dim, dim))  # Default mask
    
    # Generate a test grid to visualize conversions
    test_grid = np.zeros((dim, dim))
    
    print("\nTest Points Conversion Results:")
    print("-------------------------------")
    print(f"{'Name':<15} {'Lat':<10} {'Lon':<10} {'Grid X':<10} {'Grid Y':<10} {'In Mask?':<10}")
    print("-" * 70)
    
    for name, lat, lon, expected in test_points:
        # Forward conversion (lat/lon to grid)
        x = int(((lat_top - lat) / lat_dist) * dim)
        y = dim - int(((lon_top + abs(lon)) / lon_dist) * dim)
        
        # Handle boundary conditions
        x = max(0, min(x, dim-1))
        y = max(0, min(y, dim-1))
        
        # Mark on test grid
        test_grid[x, y] = 1
        
        # Check if point is within mask
        in_mask = mask[x, y] == 1 if mask_available else "Unknown"
        
        print(f"{name:<15} {lat:<10.4f} {lon:<10.4f} {x:<10} {y:<10} {in_mask}")
        
        # Verify against expectation
        if mask_available and expected != (mask[x, y] == 1):
            print(f"  WARNING: Expected {expected} but got {in_mask}!")
    
    # Visualize test grid with mask overlay
    plt.figure(figsize=(10, 8))
    
    # Plot mask as background
    plt.imshow(mask, cmap='gray', alpha=0.5)
    
    # Overlay test points
    for name, lat, lon, _ in test_points:
        # Convert coordinates
        x = int(((lat_top - lat) / lat_dist) * dim)
        y = dim - int(((lon_top + abs(lon)) / lon_dist) * dim)
        
        # Handle boundary conditions
        x = max(0, min(x, dim-1))
        y = max(0, min(y, dim-1))
        
        # Plot point
        plt.scatter(y, x, c='red', s=50)
        plt.text(y+5, x+5, name, fontsize=9, color='blue')
    
    plt.title('Test Points on Mask')
    plt.xlabel('Grid Y')
    plt.ylabel('Grid X')
    plt.savefig('coordinate_test.png')
    print(f"\nTest visualization saved to 'coordinate_test.png'")

def test_airnow_data(start_date, end_date, extent, save_dir='data/test_airnow.json'):
    """
    Test the AirNowData class with real data, focusing on sensor locations.
    """
    print("\n=== Testing AirNowData with Real Data ===")
    
    try:
        # Get API key
        airnow_api_key = os.environ.get('AIRNOW_API_KEY')
        if not airnow_api_key:
            print("Warning: No AirNow API key found in environment variables.")
            print("Using empty API key which may result in API request failure.")
        
        # Initialize AirNowData
        ad = AirNowData(
            start_date=start_date,
            end_date=end_date,
            extent=extent,
            airnow_api_key=airnow_api_key,
            save_dir=save_dir,
            frames_per_sample=1,  # Simple test with 1 frame per sample
            dim=200,
            idw_power=2,
            elevation_path="inputs/elevation.npy",
            mask_path="inputs/mask.npy",
            create_elevation_mask=False
        )
        
        # Check if we have sensor locations
        if not ad.air_sens_loc:
            print("No sensor locations found! This suggests there's an issue with the data or coordinate conversion.")
            return
        
        # Extract mask and sensor locations
        mask = ad.mask
        sensor_locations = ad.air_sens_loc
        
        # Print sensor locations and check against mask
        print("\nSensor Locations:")
        print("----------------")
        print(f"{'Sensor Name':<30} {'Grid X':<10} {'Grid Y':<10} {'In Mask?':<10} {'Raw Value':<10}")
        print("-" * 75)
        
        in_mask_count = 0
        outside_mask_count = 0
        
        for name, (x, y) in sensor_locations.items():
            in_mask = mask[x, y] == 1
            
            if in_mask:
                in_mask_count += 1
            else:
                outside_mask_count += 1
            
            # Get raw value from first grid if available
            raw_value = ad.ground_site_grids[0][x, y] if len(ad.ground_site_grids) > 0 else "N/A"
            
            print(f"{name:<30} {x:<10} {y:<10} {in_mask:<10} {raw_value}")
        
        print(f"\nSummary: {in_mask_count} sensors inside mask, {outside_mask_count} sensors outside mask")
        
        # Visualize sensors on mask
        visualize_sensors_on_mask(ad)
        
        # Check raw vs interpolated
        if len(ad.ground_site_grids) > 0 and len(ad.data) > 0:
            visualize_raw_vs_interpolated(ad)
            
    except Exception as e:
        print(f"Error testing AirNowData: {e}")
        import traceback
        traceback.print_exc()

def visualize_sensors_on_mask(ad):
    """
    Visualize sensor locations on the mask.
    """
    try:
        plt.figure(figsize=(12, 10))
        
        # Plot mask
        plt.imshow(ad.mask, cmap='gray', alpha=0.5)
        plt.title('Sensor Locations on Mask')
        
        # Plot sensor locations
        for name, (x, y) in ad.air_sens_loc.items():
            color = 'green' if ad.mask[x, y] == 1 else 'red'
            plt.scatter(y, x, c=color, s=50)
            plt.text(y+5, x+5, name, fontsize=8, color='blue')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Inside Mask'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outside Mask')
        ]
        plt.legend(handles=legend_elements)
        
        plt.xlabel('Grid Y')
        plt.ylabel('Grid X')
        plt.savefig('sensors_on_mask.png')
        print(f"\nSensor visualization saved to 'sensors_on_mask.png'")
        
    except Exception as e:
        print(f"Error visualizing sensors on mask: {e}")

def visualize_raw_vs_interpolated(ad):
    """
    Visualize raw vs interpolated data for the first frame.
    """
    try:
        if len(ad.ground_site_grids) == 0 or len(ad.data) == 0:
            print("No data available for visualization.")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot raw data
        im1 = axes[0].imshow(ad.ground_site_grids[0], cmap='viridis')
        axes[0].set_title('Raw Grid Data (First Frame)')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot sensor locations on raw data
        for name, (x, y) in ad.air_sens_loc.items():
            axes[0].scatter(y, x, c='red', s=40, marker='x')
            axes[0].text(y+3, x+3, name, fontsize=8, color='white')
        
        # Plot interpolated data
        im2 = axes[1].imshow(np.squeeze(ad.data[0, 0]), cmap='viridis')
        axes[1].set_title('Interpolated Data (First Frame)')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot sensor locations on interpolated data
        for name, (x, y) in ad.air_sens_loc.items():
            axes[1].scatter(y, x, c='red', s=40, marker='x')
        
        plt.tight_layout()
        plt.savefig('raw_vs_interpolated.png')
        print(f"\nRaw vs interpolated visualization saved to 'raw_vs_interpolated.png'")
        
    except Exception as e:
        print(f"Error visualizing raw vs interpolated data: {e}")

def test_extent_variations(base_extent, start_date, end_date, variations=None):
    """
    Test different variations of the extent to see which one correctly places sensors.
    """
    print("\n=== Testing Extent Variations ===")
    
    lon_bottom, lon_top, lat_bottom, lat_top = base_extent
    
    # Define variations to test if none provided
    if variations is None:
        variations = [
            # Name, Extent
            ("Original", (lon_bottom, lon_top, lat_bottom, lat_top)),
            ("Swapped Longitude", (lon_top, lon_bottom, lat_bottom, lat_top)),
            ("Swapped Latitude", (lon_bottom, lon_top, lat_top, lat_bottom)),
            ("All Swapped", (lon_top, lon_bottom, lat_top, lat_bottom)),
            ("Reordered", (lat_bottom, lat_top, lon_bottom, lon_top)),
        ]
    
    # Test each variation
    for name, extent in variations:
        print(f"\nTesting extent variation: {name}")
        print(f"Extent: {extent}")
        
        try:
            # Quick test with just the coordinate conversion
            test_coordinate_conversion(extent)
        except Exception as e:
            print(f"Error testing extent variation {name}: {e}")

def debug_single_sensor(extent, sensor_lat, sensor_lon, dim=200):
    """
    Debug the coordinate conversion for a single sensor point.
    """
    print("\n=== Debugging Single Sensor Conversion ===")
    
    # Unpack extent
    lon_bottom, lon_top, lat_bottom, lat_top = extent
    
    print(f"Extent: lon_bottom={lon_bottom}, lon_top={lon_top}, lat_bottom={lat_bottom}, lat_top={lat_top}")
    print(f"Sensor coordinates: lat={sensor_lat}, lon={sensor_lon}")
    
    # Calculate distance spans
    lat_dist = abs(lat_top - lat_bottom)
    lon_dist = abs(lon_top - lon_bottom)
    
    print(f"lat_dist = {lat_dist}, lon_dist = {lon_dist}")
    
    # Calculate x (using the formula from AirNowData)
    x = int(((lat_top - sensor_lat) / lat_dist) * dim)
    print(f"x = int(((lat_top - sensor_lat) / lat_dist) * dim)")
    print(f"x = int((({lat_top} - {sensor_lat}) / {lat_dist}) * {dim})")
    print(f"x = int(({lat_top - sensor_lat} / {lat_dist}) * {dim})")
    print(f"x = int({(lat_top - sensor_lat) / lat_dist} * {dim})")
    print(f"x = int({((lat_top - sensor_lat) / lat_dist) * dim})")
    print(f"x = {x}")
    
    # Calculate y (using the formula from AirNowData)
    y = dim - int(((lon_top + abs(sensor_lon)) / lon_dist) * dim)
    print(f"y = dim - int(((lon_top + abs(sensor_lon)) / lon_dist) * dim)")
    print(f"y = {dim} - int((({lon_top} + abs({sensor_lon})) / {lon_dist}) * {dim})")
    
    # For clarity with negative longitudes
    abs_lon = abs(sensor_lon)
    print(f"abs(sensor_lon) = {abs_lon}")
    
    print(f"y = {dim} - int((({lon_top} + {abs_lon}) / {lon_dist}) * {dim})")
    print(f"y = {dim} - int(({lon_top + abs_lon} / {lon_dist}) * {dim})")
    print(f"y = {dim} - int({(lon_top + abs_lon) / lon_dist} * {dim})")
    print(f"y = {dim} - int({((lon_top + abs_lon) / lon_dist) * dim})")
    print(f"y = {dim} - {int(((lon_top + abs_lon) / lon_dist) * dim)}")
    print(f"y = {y}")
    
    # Load mask if available
    mask_path = os.path.join('inputs', 'mask.npy')
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        in_mask = mask[x, y] == 1 if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] else False
        print(f"In mask? {in_mask}")
    else:
        print("Mask file not found, can't check if point is in mask")
    
    # Check potential coordinate issues
    if x < 0 or x >= dim or y < 0 or y >= dim:
        print("WARNING: Calculated coordinates are outside the grid boundaries!")
    
    # Test alternative conversions
    print("\nTesting alternative conversion formulas:")
    
    # Alternative 1: Different handling of longitude
    alt_y = dim - int(((lon_top - sensor_lon) / lon_dist) * dim)
    print(f"Alternative y (without abs): {alt_y}")
    
    # Alternative 2: Inverted calculation
    alt2_x = int(((sensor_lat - lat_bottom) / lat_dist) * dim)
    alt2_y = int(((sensor_lon - lon_bottom) / lon_dist) * dim)
    print(f"Alternative x,y (from bottom): {alt2_x}, {alt2_y}")

def main():
    """
    Main test function.
    """
    # Define test parameters
    # Use the same extent as in your notebook
    extent = (-118.4, 118.0, 33.9, 34.2)  # lon_bottom, lon_top, lat_bottom, lat_top
    start_date = "2025-01-10-00"
    end_date = "2025-01-13-00"
    
    print("=== AirNowData Test Script ===")
    print(f"Testing with extent: {extent}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Test basic coordinate conversion
    test_coordinate_conversion(extent)
    
    # Test extent variations
    test_extent_variations(extent, start_date, end_date)
    
    # Debug a specific sensor - replace with actual coordinates if known
    # This would be the lat/lon of a sensor that's appearing outside the mask
    debug_single_sensor(extent, 34.0, -118.2)  # Example LA coordinates
    
    # Test with real data
    test_airnow_data(start_date, end_date, extent)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()