# GHCNh Enhanced StationDataset: Pandas DatetimeIndex & Advanced Features

> **Note:** This documentation is a **work in progress** and may not fully reflect the latest code. Please refer to `ghcnh/ghcnh.py` for the most up-to-date implementation. Some features and behaviors may change as development continues.

---

This toolkit provides a powerful, user-friendly interface for working with GHCNh (Global Historical Climatology Network - Hourly) station data. It features:

- **Native pandas DatetimeIndex support** for all time operations
- **Comprehensive variable selection and filtering**
- **Robust quality control and data validation**
- **Statistical analysis and reporting**
- **Flexible data export (CSV, NetCDF)**
- **Automatic timezone handling**
- **Rich station metadata management**

---

## Key Features

### 1. Pandas DatetimeIndex Throughout

- All time coordinates are pandas `DatetimeIndex` objects (UTC by default)
- Easy time-based filtering, resampling, and timezone conversion
- No more numpy datetime64 handling

### 2. Enhanced Time Utilities

- **`get_time_index()`**: Returns the time coordinate as a pandas `DatetimeIndex`
- **`get_time_range()`**: Returns start/end time, duration, and observation count
- **`to_local_time(timezone: str)`**: Converts UTC times to any timezone
- **`add_local_time()`**: Automatically detects timezone from station coordinates (requires `timezonefinder`)

### 3. Variable Selection & Convenience Methods

- **`select_variables(variables: List[str])`**: Select variables and their metadata columns
- Convenience methods:
  - `get_temperature()`
  - `get_precipitation()`
  - `get_wind()`
  - `get_pressure()`
  - `get_humidity()`
  - `get_visibility()`
  - `get_clouds()`
  - `get_extended_precipitation()`

### 4. Quality Control

- **`apply_quality_filter(strict: bool = False)`**: *(Not yet implemented in code, see below)*
- **`clean()`**: Basic cleaning utility to handle missing values and drop rows failing QC for core variables
- **`qc_and_split_by_source_station(variable, ...)`**: Split and QC by source station for a variable

### 5. Data Analysis & Reporting

- **`get_data_availability()`**: Completeness stats for all variables
- **`get_summary_statistics(variables: Optional[List[str]])`**: Statistical summaries (count, mean, std, min, max, median, q25, q75)
- **`get_quality_report()`**: Comprehensive quality assessment (station info, data availability, summary stats, quality issues)

### 6. Station Metadata

- **`get_station_info()`**: Latitude, longitude, elevation, time range, available variables

### 7. Data Export

- **`export_to_csv(filepath, variables=None)`**: Export to CSV (all or selected variables)
- **`export_to_netcdf(filepath, variables=None)`**: Export to NetCDF
- **`to_xarray()`**: Convert to xarray Dataset
- **`to_dataframe()`**: Get pandas DataFrame
- **`to_pandas_series(variable)`**: Get pandas Series for a variable

### 8. Aggregation

- **`aggregate(variable, freq="1h", method="mean")`**: Resample a variable to a new frequency using a specified aggregation method

---

## Usage Examples

### Basic Usage

```python
from ghcnh.ghcnh import StationDataset, download_station_file

data_path = download_station_file("USW00023183", year=2023)
ds = StationDataset.from_file(data_path)

# Get time information
time_index = ds.get_time_index()
time_range = ds.get_time_range()
print(f"Dataset covers {time_range['duration_days']} days")
print(f"Timezone: {time_index.tz}")
```

### Variable Selection & Time Filtering

```python
# Get temperature data
temp_data = ds.get_temperature()
temp_series = temp_data.to_pandas_series('temperature')

# Filter for summer
summer_mask = (temp_series.index >= '2023-06-01') & (temp_series.index < '2023-09-01')
summer_temp = temp_series[summer_mask]
print(f"Summer temperature average: {summer_temp.mean():.1f}°C")
```

### Timezone Conversion

```python
# Convert to local timezone
ds_local = ds.to_local_time('America/Phoenix')
local_time = ds_local._df['local_time']
print(f"Local timezone: {local_time.dt.tz}")
```

### Data Export

```python
# Export all variables
ds.export_to_csv("station_data.csv")

# Export specific variables
temp_data = ds.get_temperature()
temp_data.export_to_csv("temperature_data.csv")
```

### Quality Control & Reporting

```python
# Clean data (handle missing, drop rows failing core QC)
clean_data = ds.clean()

# Generate quality report
report = ds.get_quality_report()
if report['quality_issues']:
    print("Quality issues detected:")
    for var, issues in report['quality_issues'].items():
        print(f"  {var}: {len(issues)} issues")
```

---

## API Reference (Work in Progress)

### Time Utilities

- **`get_time_index()`**: Returns `DatetimeIndex`
- **`get_time_range()`**: Returns dict with `start_time`, `end_time`, `duration_days`, `total_observations`
- **`to_local_time(timezone: str)`**: Returns new dataset with local time
- **`add_local_time()`**: Adds local time column (auto-detects timezone, requires `timezonefinder`)

### Variable Selection

- **`select_variables(variables: List[str])`**: Selects variables and metadata columns
- **Convenience methods**:
  - `get_temperature()`, `get_precipitation()`, `get_wind()`, `get_pressure()`, `get_humidity()`, `get_visibility()`, `get_clouds()`, `get_extended_precipitation()`

### Quality Control

- **`clean()`**: Cleans dataset by handling missing values and dropping rows failing QC for core variables
- **`qc_and_split_by_source_station(variable, ...)`**: Splits and QCs by source station for a variable
- **`apply_quality_filter(strict: bool = False)`**: *(Not yet implemented in code, see below)*

### Data Analysis

- **`get_data_availability()`**: Returns completeness stats for each variable
- **`get_summary_statistics(variables: Optional[List[str]])`**: Returns stats (count, mean, std, min, max, median, q25, q75)
- **`get_quality_report()`**: Returns dict with station info, data availability, summary stats, and quality issues

### Station Metadata

- **`get_station_info()`**: Returns dict with latitude, longitude, elevation, time range, available variables

### Export Methods

- **`export_to_csv(filepath, variables=None)`**: Exports to CSV
- **`export_to_netcdf(filepath, variables=None)`**: Exports to NetCDF
- **`to_xarray()`**: Convert to xarray Dataset
- **`to_dataframe()`**: Get pandas DataFrame
- **`to_pandas_series(variable)`**: Get pandas Series for a variable

### Aggregation

- **`aggregate(variable, freq="1h", method="mean")`**: Resample a variable to a new frequency using a specified aggregation method

---

## Available Variables

- **Core**: `temperature`, `dew_point_temperature`, `wind_speed`, `wind_direction`, `precipitation`, `solar_radiation`
- **Extended**: `relative_humidity`, `station_level_pressure`, `visibility`, `wet_bulb_temperature`, `wind_gust`, `snow_depth`, `sea_level_pressure`, `sky_cover_1/2/3`, `sky_cover_baseht_1/2/3`, `precipitation_3h/6h/12h/24h`, `precipitation_estimated`, `solar_radiation_estimated`, `temperature_max/min`, `dew_point_temperature_max/min`, `relative_humidity_max/min`, `wind_speed_max/min`, `station_level_pressure_max/min`, `sea_level_pressure_max/min`

---

## Time-based Operations

- **Filtering**:
  ```python
  jan_mask = (time_index >= '2023-01-01') & (time_index < '2023-02-01')
  jan_data = temp_series[jan_mask]
  ```
- **Resampling**:
  ```python
  daily_temp = temp_series.resample('D').mean()
  hourly_temp = temp_series.resample('H').mean()
  ```
- **Timezone conversion**:
  ```python
  local_time = time_index.tz_convert('America/New_York')
  ```

---

## Quality Control Details (Work in Progress)

- **Good QC Codes:**
  - See `_GOOD_QC_CODES` in `ghcnh.py` (currently only used in `qc_and_split_by_source_station`)
- **Bad QC Codes:**
  - See `_BAD_QC` in `ghcnh.py`
- **`clean()`**: Drops rows failing QC for core variables, replaces missing/trace values
- **`qc_and_split_by_source_station()`**: Masks bad QC for a variable, splits by source station
- **`apply_quality_filter()`**: *Not yet implemented in code*

---

## Cleaning vs. Quality Filtering: `clean()` vs. `apply_quality_filter()`

- **`clean()`**: Handles missing values, sets trace precipitation to 0, drops rows failing QC for core variables
- **`apply_quality_filter()`**: *Not yet implemented in code*
- **`qc_and_split_by_source_station()`**: For a variable, splits by source station, masks bad QC, drops NaN, returns dict of StationDataset

---

## Requirements

- pandas >= 1.3.0
- xarray >= 0.19.0
- numpy >= 1.20.0
- requests >= 2.25.0
- tqdm >= 4.60.0
- timezonefinder (optional, for automatic timezone detection)

---

## File Structure

- `ghcnh/ghcnh.py` - Main code with enhanced features and pandas DatetimeIndex support
- `ghcnh/example_pandas_datetime.py` - Example usage script *(not present, update as needed)*
- `ghcnh/README.md` - This documentation

---

## Benefits

- **Easier Time Manipulation**: Native pandas datetime support
- **Better Timezone Support**: Automatic and manual timezone handling
- **Intuitive Filtering**: Use pandas boolean indexing with datetime strings
- **Built-in Resampling**: Easy aggregation to different time frequencies
- **Better Integration**: Works seamlessly with pandas/xarray ecosystem
- **Type Safety**: Proper type hints and error handling
- **Comprehensive Quality Control**: Robust QC filtering and reporting (WIP)
- **Flexible Export**: CSV and NetCDF support

---

## Error Handling

- Invalid variable names are logged as warnings
- Missing quality codes are handled gracefully
- Export functions create directories as needed
- Statistical functions handle empty datasets

---

## Example: End-to-End Workflow

```python
from ghcnh.ghcnh import StationDataset, download_station_file

# Download and load data
station_id = "USW00023183"
data_path = download_station_file(station_id, year=2023)
ds = StationDataset.from_file(data_path)

# Get station info
info = ds.get_station_info()
print(f"Station: {info['latitude']}°N, {info['longitude']}°W")

# Select and analyze temperature data
temp_data = ds.get_temperature()
temp_stats = temp_data.get_summary_statistics()
print(f"Temperature mean: {temp_stats['temperature']['mean']:.2f}°C")

# Clean data (handle missing, drop rows failing core QC)
clean_data = ds.clean()

# Export filtered data
clean_data.export_to_csv("clean_station_data.csv")

# Generate quality report
report = ds.get_quality_report()
if report['quality_issues']:
    print("Quality issues detected:")
    for var, issues in report['quality_issues'].items():
        print(f"  {var}: {len(issues)} issues")
```

---

For more details, see the code and examples in the repository. 
