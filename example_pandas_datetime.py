#!/usr/bin/env python3
"""
Example usage of the updated GHCNh StationDataset class with pandas DatetimeIndex

This script demonstrates the improved time handling with pandas DatetimeIndex
instead of numpy arrays for easier datetime manipulation.
"""

import pathlib
from ghcnh_gpt_updated import StationDataset, download_station_file

def main():
    """Demonstrate the pandas DatetimeIndex functionality."""
    
    # Example station ID (Phoenix Airport)
    station_id = "USW00023183"
    
    print(f"Downloading data for station {station_id}...")
    
    try:
        # Download station data for 2023
        data_path = download_station_file(station_id, year=2023)
        
        # Load the data
        ds = StationDataset.from_file(data_path)
        print(f"Loaded dataset with {len(ds.get_time_index())} observations")
        
        # Demonstrate time handling
        print("\n=== Time Handling ===")
        time_range = ds.get_time_range()
        print(f"Time range: {time_range['start_time']} to {time_range['end_time']}")
        print(f"Duration: {time_range['duration_days']} days")
        print(f"Total observations: {time_range['total_observations']}")
        
        # Get time index as pandas DatetimeIndex
        time_index = ds.get_time_index()
        print(f"\nTime index type: {type(time_index)}")
        print(f"Time index timezone: {time_index.tz}")
        print(f"First 5 timestamps: {time_index[:5]}")
        
        # Demonstrate timezone conversion
        print("\n=== Timezone Conversion ===")
        try:
            # Convert to local time (Phoenix timezone)
            ds_local = ds.to_local_time('America/Phoenix')
            local_time_index = ds_local._ds.local_time.values
            print(f"Local time index type: {type(local_time_index)}")
            print(f"Local time timezone: {local_time_index.tz}")
            print(f"First 5 local timestamps: {local_time_index[:5]}")
        except Exception as e:
            print(f"Timezone conversion failed: {e}")
        
        # Demonstrate variable selection with pandas Series
        print("\n=== Variable Selection ===")
        temp_data = ds.get_temperature()
        temp_series = temp_data.to_pandas_series('temperature')
        print(f"Temperature series type: {type(temp_series)}")
        print(f"Temperature series index type: {type(temp_series.index)}")
        print(f"Temperature data (first 5):\n{temp_series.head()}")
        
        # Demonstrate DataFrame export with DatetimeIndex
        print("\n=== DataFrame Export ===")
        df = ds.to_dataframe()
        print(f"DataFrame index type: {type(df.index)}")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame head:\n{df.head()}")
        
        # Demonstrate time-based filtering
        print("\n=== Time-based Filtering ===")
        # Filter for January 2023
        jan_mask = (time_index >= '2023-01-01') & (time_index < '2023-02-01')
        jan_data = temp_series[jan_mask]
        print(f"January 2023 temperature data: {len(jan_data)} observations")
        print(f"January temperature stats: mean={jan_data.mean():.1f}°C, "
              f"min={jan_data.min():.1f}°C, max={jan_data.max():.1f}°C")
        
        # Demonstrate resampling
        print("\n=== Time Resampling ===")
        daily_temp = temp_series.resample('D').mean()
        print(f"Daily temperature data: {len(daily_temp)} days")
        print(f"Daily temperature (first 5):\n{daily_temp.head()}")
        
        # Export with proper DatetimeIndex
        output_dir = pathlib.Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Export temperature data to CSV with DatetimeIndex
        temp_data.export_to_csv(str(output_dir / f"{station_id}_temperature_2023.csv"))
        print(f"\nExported temperature data to {output_dir / f'{station_id}_temperature_2023.csv'}")
        
        # Export full dataset to CSV
        ds.export_to_csv(str(output_dir / f"{station_id}_full_2023.csv"))
        print(f"Exported full dataset to {output_dir / f'{station_id}_full_2023.csv'}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 