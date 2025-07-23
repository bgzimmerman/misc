# GHCNh Enhanced StationDataset: Pandas DatetimeIndex & Advanced Features

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
- **`add_local_time()`**: Automatically detects timezone from station coordinates

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

- **`apply_quality_filter(strict: bool = False)`**: Filter data using GHCNh QC codes
  - Permissive (default): keeps data not clearly bad
  - Strict: only keeps clearly good data
- **`clean()`**: Basic cleaning utility to handle missing values and drop rows failing QC for core variables

### 5. Data Analysis & Reporting

- **`get_data_availability()`**: Completeness stats for all variables
- **`get_summary_statistics(variables: Optional[List[str]])`**: Statistical summaries (count, mean, std, min, max, median, q25, q75)
- **`get_quality_report()`**: Comprehensive quality assessment (station info, data availability, summary stats, quality issues)

### 6. Station Metadata

- **`get_station_info()`**: Latitude, longitude, elevation, time range, available variables

### 7. Data Export

- **`export_to_csv(filepath, variables=None)`**: Export to CSV (all or selected variables)
- **`export_to_netcdf(filepath, variables=None)`**: Export to NetCDF

---

## Usage Examples

### Basic Usage

```python
from ghcnh.ghcnh import StationDataset, download_station_file

# Download and load data
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
# Apply quality filtering
clean_data = ds.apply_quality_filter(strict=False)

# Generate quality report
report = ds.get_quality_report()
if report['quality_issues']:
    print("Quality issues detected:")
    for var, issues in report['quality_issues'].items():
        print(f"  {var}: {len(issues)} issues")
```

---

## API Reference

### Time Utilities

- **`get_time_index()`**: Returns `DatetimeIndex`
- **`get_time_range()`**: Returns dict with `start_time`, `end_time`, `duration_days`, `total_observations`
- **`to_local_time(timezone: str)`**: Returns new dataset with local time
- **`add_local_time()`**: Adds local time column (auto-detects timezone)

### Variable Selection

- **`select_variables(variables: List[str])`**: Selects variables and metadata columns
- **Convenience methods**:  
  - `get_temperature()`, `get_precipitation()`, `get_wind()`, `get_pressure()`, `get_humidity()`, `get_visibility()`, `get_clouds()`, `get_extended_precipitation()`

### Quality Control

- **`apply_quality_filter(strict: bool = False)`**: Filters data by QC codes
  - **Strict mode:** Only numeric codes `0`, `1`, `4`, `5`, `9` and legacy codes `313`, `346` are accepted; any letter code is considered bad and masked.
  - **Permissive mode:** Numeric codes `0` through `8` and legacy codes `313`, `346` are accepted (legacy, more lenient behavior).
- **`clean()`**: Cleans dataset by handling missing values and dropping rows failing QC for core variables

### Data Analysis

- **`get_data_availability()`**: Returns completeness stats for each variable
- **`get_summary_statistics(variables: Optional[List[str]])`**: Returns stats (count, mean, std, min, max, median, q25, q75)
- **`get_quality_report()`**: Returns dict with station info, data availability, summary stats, and quality issues

### Station Metadata

- **`get_station_info()`**: Returns dict with latitude, longitude, elevation, time range, available variables

### Export Methods

- **`export_to_csv(filepath, variables=None)`**: Exports to CSV
- **`export_to_netcdf(filepath, variables=None)`**: Exports to NetCDF

### Aggregation

- **`aggregate(freq: str = "1h")`**: Resample to a new frequency using variable-aware aggregation rules

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

## Quality Control Details

- **Good QC Codes (as used in code):**
  - **Strict mode:** Only numeric codes `0`, `1`, `4`, `5`, `9` (meaning "passed" or "not flagged" per GHCNh documentation) and legacy codes `313`, `346` are accepted. **Any letter code (e.g., 'L', 'F', etc.) is considered bad and will be masked.**
  - **Permissive mode:** Numeric codes `0` through `8` and legacy codes `313`, `346` are accepted (legacy behavior, more lenient).
- **Bad QC Codes (filtered by default):**
  - Numeric codes `2`, `3`, `6`, `7` (suspect/erroneous per docs)
  - Any letter code (e.g., `L`, `F`, `K`, etc.) means a specific check failed and is considered bad
- The `apply_quality_filter` method uses these codes to mask or drop bad data. The `clean()` method drops rows failing QC for core variables.
- **Note:** The strict mode is now fully aligned with the official GHCNh documentation for maximum data reliability.

---

## Cleaning vs. Quality Filtering: `clean()` vs. `apply_quality_filter()`

The toolkit provides two main methods for data cleaning and quality control:

### `clean()`
- **Purpose:** Broad, basic cleaning of the dataset.
- **Actions:**
  - Replaces sentinel missing values (e.g., -9999, 9999, empty string) with `np.nan` for all numeric columns.
  - Sets trace precipitation values (e.g., 'T', 'Trace') to 0.0 in the `precipitation` column.
  - **Drops entire rows** where any *core variable* (temperature, dew point, wind speed, wind direction, precipitation, solar radiation) has a "bad" QC code.
- **Result:** Returns a new dataset with all rows failing core variable QC removed, and missing/trace values handled.

### `apply_quality_filter(strict: bool = False)`
- **Purpose:** Flexible, per-variable quality control without dropping rows.
- **Actions:**
  - For each variable with a QC column:
    - **Strict mode:** Only accepts numeric QC codes `0`, `1`, `4`, `5`, `9` and legacy codes `313`, `346` (per official documentation). **Any letter code is considered bad and masked.**
    - **Permissive mode:** Accepts numeric codes `0` through `8` and legacy codes `313`, `346` (legacy, more lenient behavior).
    - For any value not in the "good" list, that variable’s value is set to `np.nan` for those observations.
  - **Does not drop rows**—just masks out “bad” values for each variable independently.
  - The `strict` flag controls how selective the filter is.
- **Result:** Returns a new dataset where “bad” values are masked (set to `np.nan`), but the structure and length of the dataset is preserved.

### Summary Table

| Method                | Handles Missing/Trace | Drops Rows | Masks Bad Values | Per-Variable Control | QC Strictness Option |
|-----------------------|----------------------|-----------|------------------|---------------------|----------------------|
| `clean()`             | Yes                  | Yes       | No               | No                  | No                   |
| `apply_quality_filter`| No                   | No        | Yes              | Yes                 | Yes (`strict`)       |

### When to Use Which?
- **Use `clean()`** when you want a quick, broad cleaning: remove all rows with any core variable failing QC, and handle missing/trace values.
- **Use `apply_quality_filter()`** when you want to keep the full dataset but mask out only the “bad” values for each variable, with optional strictness.
- **For maximum reliability, use `apply_quality_filter(strict=True)` to ensure only the highest-quality data is retained, as defined by the official GHCNh documentation.**

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
- `ghcnh/example_pandas_datetime.py` - Example usage script
- `ghcnh/README.md` - This documentation

---

## Benefits

- **Easier Time Manipulation**: Native pandas datetime support
- **Better Timezone Support**: Automatic and manual timezone handling
- **Intuitive Filtering**: Use pandas boolean indexing with datetime strings
- **Built-in Resampling**: Easy aggregation to different time frequencies
- **Better Integration**: Works seamlessly with pandas/xarray ecosystem
- **Type Safety**: Proper type hints and error handling
- **Comprehensive Quality Control**: Robust QC filtering and reporting
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

# Apply quality filtering
clean_data = ds.apply_quality_filter(strict=False)

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
