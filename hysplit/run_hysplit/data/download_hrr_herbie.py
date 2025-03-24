#!/usr/bin/env python3
import os
import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor
import sys
import time

def download_hrrr_smoke(date, hour, fhr, output_dir, product="hrrr", field="sfc"):
    """
    Download HRRR data with smoke variables using Herbie
    
    Parameters:
    -----------
    date : datetime.datetime
        Date of the model run
    hour : int
        Hour of the model run (0-23)
    fhr : int
        Forecast hour
    output_dir : str
        Directory to save downloaded files
    product : str
        HRRR product ("hrrr" or "hrrr_smoke")
    field : str
        Field to download (e.g., "sfc" for surface fields)
    """
    try:
        # Only import Herbie here to avoid issues if it's not installed
        from herbie import Herbie
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Format the date and time
        date_str = date.strftime("%Y%m%d")
        
        # Initialize Herbie
        H = Herbie(
            date=date_str,
            model=product,
            product=field,  # Use "sfc" for surface fields
            fxx=fhr,
            run=hour
        )
        
        # Set the output path
        # Format: hrrr.YYYYMMDD.HHz.fFFh.field.grib2
        outfile = os.path.join(
            output_dir, 
            f"{product}.{date_str}.{hour:02d}z.f{fhr:02d}h.{field}.grib2"
        )
        
        # Check if file already exists
        if os.path.exists(outfile) and os.path.getsize(outfile) > 1000000:
            print(f"File already exists: {os.path.basename(outfile)}")
            return True
        
        print(f"Downloading: {os.path.basename(outfile)}")
        
        # Download the data
        # For smoke variables, we need to search for specific variables
        # MASSDEN = Smoke Mass Density
        # COLMD = Column-Integrated Smoke
        # AOTK = Aerosol Optical Thickness
        
        # List of variables for smoke concentration
        smoke_vars = ["MASSDEN", "COLMD", "AOTK", "SMOKE"]
        
        try:
            # Try to download with specific search for smoke variables first
            H.download(outfile, searchString="|".join(smoke_vars))
            
            # Verify the download was successful and contains data
            if os.path.exists(outfile) and os.path.getsize(outfile) > 10000:
                print(f"Successfully downloaded smoke data: {os.path.basename(outfile)}")
                return True
                
        except Exception as e:
            print(f"Could not find specific smoke variables, trying full file: {str(e)}")
        
        # If specific variables failed, try downloading the complete file
        try:
            H.download(outfile)
            
            # Verify the download was successful
            if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                print(f"Successfully downloaded full file: {os.path.basename(outfile)}")
                return True
            else:
                print(f"Error: Downloaded file is empty or doesn't exist")
                return False
                
        except Exception as e:
            print(f"Failed to download full file: {str(e)}")
            return False
            
    except ImportError:
        print("Error: Herbie is not installed. Please install with: pip install herbie-data")
        return False
        
    except Exception as e:
        print(f"Error downloading {product} data for {date_str} {hour:02d}Z F{fhr:02d}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download HRRR Smoke data using Herbie')
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
    parser.add_argument('--output-dir', type=str, default='./herbie_smoke_data', 
                        help='Output directory')
    parser.add_argument('--product', type=str, default='hrrr', choices=['hrrr', 'hrrr_smoke'],
                        help='HRRR product to download (default: hrrr)')
    parser.add_argument('--field', type=str, default='sfc', choices=['sfc', 'prs', 'nat', 'subh'],
                        help='Field to download (default: sfc - surface fields)')
    parser.add_argument('--threads', type=int, default=2, 
                        help='Number of concurrent downloads (default: 2)')
    
    args = parser.parse_args()
    
    # Validate date
    try:
        date = datetime.datetime.strptime(args.date, '%Y%m%d')
    except ValueError:
        print("Invalid date format. Please use YYYYMMDD")
        return 1
    
    # Create list of forecast hours
    forecast_hours = list(range(args.fhr_start, args.fhr_end + 1, args.fhr_step))
    
    print(f"HRRR Smoke Data Downloader using Herbie")
    print(f"======================================")
    print(f"Date: {date.strftime('%Y-%m-%d')}")
    print(f"Model runs: {args.hour}")
    print(f"Forecast hours: {forecast_hours}")
    print(f"Product: {args.product}")
    print(f"Field: {args.field}")
    print(f"Output directory: {args.output_dir}")
    print(f"Concurrent downloads: {args.threads}")
    print(f"======================================")
    
    # Check if Herbie is installed
    try:
        import herbie
        print(f"Herbie version: {herbie.__version__}")
    except ImportError:
        print("Herbie is not installed. Please install with: pip install herbie-data")
        print("Would you like to install it now? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            import subprocess
            print("Installing Herbie...")
            subprocess.run([sys.executable, "-m", "pip", "install", "herbie-data"])
            print("Herbie installed. Please run the script again.")
        return 1
    
    # Create a download queue
    download_queue = []
    for hour in args.hour:
        for fhr in forecast_hours:
            download_queue.append((date, hour, fhr, args.output_dir, args.product, args.field))
    
    total_files = len(download_queue)
    print(f"Preparing to download {total_files} file(s)...")
    
    # Download files concurrently
    successful = 0
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(executor.map(lambda params: download_hrrr_smoke(*params), download_queue))
        successful = sum(1 for r in results if r)
    
    print(f"Download summary: {successful} of {total_files} files successfully downloaded")
    
    if successful < total_files:
        print(f"Warning: {total_files - successful} file(s) failed to download")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())