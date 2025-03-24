#!/usr/bin/env python3
import os
import requests
import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
import sys
import subprocess

def download_hrrr_full_file(date, hour, forecast_hour, output_dir, retry=3):
    """
    Download full HRRR file and then extract California region
    
    Parameters:
    -----------
    date : datetime.datetime
        Date of the model run
    hour : int
        Hour of the model run (0-23)
    forecast_hour : int
        Forecast hour to download
    output_dir : str
        Directory to save the downloaded file
    retry : int
        Number of retry attempts for failed downloads
    """
    
    # AWS S3 HRRR archive is often more reliable than NOMADS
    base_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date}/conus/hrrr.t{hour}z.wrfsfcf{fhr}.grib2"
    
    date_str = date.strftime("%Y%m%d")
    hour_str = f"{hour:02d}"
    fhr_str = f"{forecast_hour:02d}"
    
    # Create formatted URL
    url = base_url.format(date=date_str, hour=hour_str, fhr=fhr_str)
    
    # Output filenames
    full_filename = f"hrrr.t{hour_str}z.wrfsfcf{fhr_str}.grib2"
    full_output_file = os.path.join(output_dir, full_filename)
    
    ca_filename = f"hrrr.t{hour_str}z.wrfsfcf{fhr_str}.california.grib2"
    ca_output_file = os.path.join(output_dir, ca_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the file with retries
    attempts = 0
    while attempts < retry:
        try:
            print(f"Downloading full HRRR file: {full_filename} (Attempt {attempts+1}/{retry})")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            file_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(full_output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        status = f"Downloaded: {downloaded/1024/1024:.1f} MB / {file_size/1024/1024:.1f} MB"
                        print(status, end='\r')
            
            print(f"\nSuccessfully downloaded: {full_filename}")
            
            # Verify file size
            if file_size > 0 and os.path.getsize(full_output_file) < file_size * 0.9:
                print(f"Warning: File {full_filename} may be incomplete")
                attempts += 1
                continue
            
            # Extract California region using wgrib2 if available
            try:
                # California bounding box
                lat1, lon1 = 32.5, -124.6  # SW corner
                lat2, lon2 = 42.0, -114.0  # NE corner
                
                if_wgrib2_exists = subprocess.run(["which", "wgrib2"], 
                                                 stdout=subprocess.PIPE, 
                                                 stderr=subprocess.PIPE).returncode == 0
                
                if if_wgrib2_exists:
                    print(f"Extracting California region to {ca_filename}")
                    cmd = [
                        "wgrib2", full_output_file,
                        "-small_grib", f"{lon1}:{lon2}", f"{lat1}:{lat2}",
                        ca_output_file
                    ]
                    subprocess.run(cmd, check=True)
                    print(f"Successfully extracted California region")
                    
                    # Optionally remove the full file to save space
                    # os.remove(full_output_file)
                else:
                    print("wgrib2 not found. Keeping full file only.")
            
            except subprocess.SubprocessError as e:
                print(f"Warning: Failed to extract California region: {e}")
                print(f"Keeping full file at {full_output_file}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading (attempt {attempts+1}): {e}")
            attempts += 1
            if attempts < retry:
                wait_time = 30 * attempts  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download after {retry} attempts")
                return False

def main():
    parser = argparse.ArgumentParser(description='Download HRRR files (full files with optional California extraction)')
    parser.add_argument('--date', type=str, 
                        default=datetime.datetime.now().strftime('%Y%m%d'),
                        help='Date in YYYYMMDD format (default: today)')
    parser.add_argument('--hour', type=int, nargs='+',
                        default=[0, 6, 12, 18],
                        help='Model run hour(s) (0-23), can specify multiple (default: 0 6 12 18)')
    parser.add_argument('--fhr-start', type=int, default=0, 
                        help='Start forecast hour (default: 0)')
    parser.add_argument('--fhr-end', type=int, default=18, 
                        help='End forecast hour (default: 18)')
    parser.add_argument('--fhr-step', type=int, default=1, 
                        help='Forecast hour step (default: 1 - download every hour)')
    parser.add_argument('--output-dir', type=str, default='./hrrr_data', 
                        help='Output directory')
    parser.add_argument('--threads', type=int, default=2, 
                        help='Number of concurrent downloads (default: 2)')
    parser.add_argument('--retry', type=int, default=3,
                        help='Number of retry attempts (default: 3)')
    
    args = parser.parse_args()
    
    # Validate date
    try:
        date = datetime.datetime.strptime(args.date, '%Y%m%d')
    except ValueError:
        print("Invalid date format. Please use YYYYMMDD")
        return 1
    
    # Validate hours
    for hour in args.hour:
        if hour < 0 or hour > 23:
            print(f"Error: Invalid hour {hour}. Hours must be between 0-23.")
            return 1
    
    # Validate forecast hour range
    if args.fhr_start < 0 or args.fhr_end < args.fhr_start:
        print("Error: Invalid forecast hour range.")
        return 1
    
    # Create list of forecast hours
    forecast_hours = list(range(args.fhr_start, args.fhr_end + 1, args.fhr_step))
    
    print(f"HRRR Direct File Downloader")
    print(f"=========================")
    print(f"Date: {date.strftime('%Y-%m-%d')}")
    print(f"Model runs: {args.hour}")
    print(f"Forecast hours: {forecast_hours}")
    print(f"Output directory: {args.output_dir}")
    print(f"Concurrent downloads: {args.threads}")
    print(f"=========================")
    
    # Create a download queue
    download_queue = []
    for hour in args.hour:
        for fhr in forecast_hours:
            download_queue.append((date, hour, fhr, args.output_dir, args.retry))
    
    total_files = len(download_queue)
    print(f"Preparing to download {total_files} file(s)...")
    
    # Download files concurrently
    successful = 0
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(executor.map(lambda params: download_hrrr_full_file(*params), download_queue))
        successful = sum(1 for r in results if r)
    
    print(f"Download summary: {successful} of {total_files} files successfully downloaded")
    if successful < total_files:
        print(f"Warning: {total_files - successful} file(s) failed to download")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())