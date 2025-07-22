# Enhanced GHCNh StationDataset Features

This document describes the new utilities and features added to the `StationDataset` class in `ghcnh_gpt.py`.

## Overview

The enhanced `StationDataset` class now provides comprehensive utilities for:
- Variable selection and filtering
- Quality control and data validation
- Statistical analysis and reporting
- Data export and visualization
- Station metadata management

## New Features

### 1. Variable Selection

#### `select_variables(variables: List[str]) -> StationDataset`
Select specific variables and their associated metadata columns.

```python
# Select temperature and precipitation data
temp_precip = ds.select_variables(['temperature', 'precipitation'])

# The returned dataset includes:
# - temperature, temperature_Quality_Code, temperature_Measurement_Code, etc.
# - precipitation, precipitation_Quality_Code, precipitation_Measurement_Code, etc.
# - time, LATITUDE, LONGITUDE, ELEVATION coordinates
```

#### Convenience Methods
```python
# Get specific variable groups
temp_only = ds.get_temperature()
precip_only = ds.get_precipitation()
wind_data = ds.get_wind()
pressure_data = ds.get_pressure()
humidity_data = ds.get_humidity()
visibility_data = ds.get_visibility()
cloud_data = ds.get_clouds()
extended_precip = ds.get_extended_precipitation()  # All precipitation variables
```

### 2. Quality Control

#### `apply_quality_filter(strict: bool = False) -> StationDataset`
Apply quality control filtering to the dataset.

```python
# Permissive QC (default) - keeps data that's not clearly bad
clean_data = ds.apply_quality_filter(strict=False)

# Strict QC - only keeps clearly good data
strict_clean_data = ds.apply_quality_filter(strict=True)
```

The quality control uses the GHCNh documentation's QC codes:
- **General codes**: 0, 1, 4, 5, 9, A, C, R, None, "9-Missing"
- **Legacy codes**: Additional codes from sources 313-346
- **Source-specific codes**: Special handling for sources 382, 345

### 3. Data Analysis

#### `get_data_availability() -> Dict[str, Dict[str, Any]]`
Get comprehensive data availability statistics.

```python
availability = ds.get_data_availability()

for var, stats in availability.items():
    print(f"{var}: {stats['completeness_percent']:.1f}% complete")
    print(f"  Total observations: {stats['total_observations']}")
    print(f"  Valid observations: {stats['valid_observations']}")
    print(f"  Units: {stats['units']}")
    print(f"  Description: {stats['description']}")
```

#### `get_summary_statistics(variables: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]`
Get statistical summaries for variables.

```python
# Get stats for all available variables
all_stats = ds.get_summary_statistics()

# Get stats for specific variables
temp_stats = ds.get_summary_statistics(['temperature', 'precipitation'])

# Available statistics: count, mean, std, min, max, median, q25, q75
```

#### `get_time_range() -> Dict[str, Any]`
Get time range information.

```python
time_info = ds.get_time_range()
print(f"Start: {time_info['start_time']}")
print(f"End: {time_info['end_time']}")
print(f"Duration: {time_info['duration_days']} days")
print(f"Total observations: {time_info['total_observations']}")
```

### 4. Station Information

#### `get_station_info() -> Dict[str, Any]`
Get comprehensive station metadata.

```python
station_info = ds.get_station_info()
print(f"Latitude: {station_info['latitude']}")
print(f"Longitude: {station_info['longitude']}")
print(f"Elevation: {station_info['elevation']} m")
print(f"Variables available: {station_info['variables_available']}")
```

### 5. Quality Reporting

#### `get_quality_report() -> Dict[str, Any]`
Generate a comprehensive quality assessment report.

```python
report = ds.get_quality_report()

# Report includes:
# - station_info: Station metadata
# - data_availability: Data completeness stats
# - summary_statistics: Statistical summaries
# - quality_issues: Identified quality problems
```

### 6. Data Export

#### `export_to_csv(filepath: str, variables: Optional[List[str]] = None) -> None`
Export data to CSV format.

```python
# Export all variables
ds.export_to_csv("station_data.csv")

# Export specific variables
ds.export_to_csv("temp_precip.csv", variables=['temperature', 'precipitation'])
```

#### `export_to_netcdf(filepath: str, variables: Optional[List[str]] = None) -> None`
Export data to NetCDF format.

```python
# Export all variables
ds.export_to_netcdf("station_data.nc")

# Export specific variables
ds.export_to_netcdf("temp_precip.nc", variables=['temperature', 'precipitation'])
```

## Available Variables

The enhanced class supports all GHCNh variables:

### Core Variables
- `temperature` - 2 meter air temperature (°C to tenths)
- `dew_point_temperature` - Dew point temperature (°C to tenths)
- `wind_speed` - Wind speed (m/s)
- `wind_direction` - Wind direction from true north (degrees)
- `precipitation` - Total liquid precipitation (mm)
- `relative_humidity` - Relative humidity (percent)
- `station_level_pressure` - Station pressure (hPa)
- `visibility` - Horizontal visibility (km)
- `wet_bulb_temperature` - Wet bulb temperature (°C to tenths)

### Extended Variables
- `wind_gust` - Peak wind gust (m/s)
- `snow_depth` - Snow depth (mm)
- `altimeter` - Altimeter pressure (hPa)
- `pressure_3hr_change` - 3-hour pressure change (hPa)
- `sky_cover_1/2/3` - Sky cover layers (oktas)
- `sky_cover_baseht_1/2/3` - Cloud base heights (m)
- `precipitation_3hour/6hour/9hour/12hour/15hour/18hour/21hour/24hour` - Extended precipitation totals (mm)

## Example Usage

```python
from ghcnh_gpt import StationDataset, download_station_file

# Download and load data
station_id = "USW00023183"  # Phoenix Airport
data_path = download_station_file(station_id, year=2023)
ds = StationDataset.from_file(data_path)

# Get station information
station_info = ds.get_station_info()
print(f"Station: {station_info['latitude']}°N, {station_info['longitude']}°W")

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

## Quality Control Details

The enhanced quality control system handles multiple QC code schemes:

1. **General QC Codes** (applied to integrated sources):
   - 0, 1, 4, 5, 9, A, C, R, None, "9-Missing" = Good data
   - L, o, F, U, D, d, W, K, C, T, S, h, V, w, N, E, p, H = Bad data

2. **Legacy QC Codes** (sources 313-346):
   - 0, 1, 4, 5, 9, A, U, P, I, M, C, R = Good data
   - 2, 3, 6, 7 = Bad data

3. **Source-specific codes**:
   - Source 382: Blank, A, M, D, Q, q, R = Good data
   - Source 345: 0 = Good data

## Data Export Formats

### CSV Export
- Includes all selected variables and their metadata columns
- Time index as first column
- Station coordinates included
- Quality codes and measurement codes preserved

### NetCDF Export
- Preserves xarray dataset structure
- Includes all coordinates and attributes
- Compatible with climate data analysis tools

## Performance Considerations

- Variable selection creates new datasets but preserves original data
- Quality filtering applies masks without copying data
- Statistical calculations are computed on-demand
- Large datasets may benefit from chunked processing

## Error Handling

The enhanced class includes comprehensive error handling:
- Invalid variable names are logged as warnings
- Missing quality codes are handled gracefully
- Export functions create directories as needed
- Statistical functions handle empty datasets

## Dependencies

The enhanced features require:
- pandas >= 1.3.0
- xarray >= 0.19.0
- numpy >= 1.20.0
- requests >= 2.25.0

Optional dependencies:
- timezonefinder (for local time conversion)
- tqdm (for progress bars) 