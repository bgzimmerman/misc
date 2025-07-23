import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm.auto import tqdm

try:
    from timezonefinder import TimezoneFinder
except ImportError:
    TimezoneFinder = None

_logger = logging.getLogger("ghcnh_toolkit")

__all__ = [
    "read_station_catalog",
    "download_station_file",
    "open_ghcnh_dataframe",
    "StationDataset",
]

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# canonical sentinel / trace codes
_MISSING_VALUES = {-9999, -9999.9, -999, -999.9, 9999, 9999.9, "", np.nan}

# Measurement codes for trace precipitation
_TRACE_CODES = {"T", "Trace", "trace", "2-Trace", "2"}

# Good QC codes by source
_GOOD_QC_CODES = {
    "source_code_set_1": ["0", "1", "4", "5", "9", "A", "U", "P", "I", "M", "C", "R"],
}



# QC values considered *bad* – will be filtered (updated for strictness)
# Any letter code means a specific check failed and is considered bad
_BAD_QC = {
    # Numeric codes that mean suspect or erroneous
    "2", "3", "6", "7",  # suspect/erroneous (per docs)
    # Letter codes for failed checks (from documentation Table 3)
    "L", "o", "F", "U", "D", "d", "W", "K", "C", "T", "S", "h", "V", "w", "N", "E", "p", "H"
}

# variable-specific aggregation semantics
_CORE_VARS: Dict[str, str] = {
    "temperature": "mean",
    "dew_point_temperature": "mean",
    "wind_speed": "mean",
    "wind_direction": "mean",
    "precipitation": "sum",
    "solar_radiation": "mean",
}

# Available variables in GHCNh data with their quality code suffixes
_AVAILABLE_VARS = {
    "temperature": "temperature_Quality_Code",
    "dew_point_temperature": "dew_point_temperature_Quality_Code", 
    "wind_speed": "wind_speed_Quality_Code",
    "wind_direction": "wind_direction_Quality_Code",
    "precipitation": "precipitation_Quality_Code",
    "relative_humidity": "relative_humidity_Quality_Code",
    "station_level_pressure": "station_level_pressure_Quality_Code",
    "visibility": "visibility_Quality_Code",
    "wet_bulb_temperature": "wet_bulb_temperature_Quality_Code",
    "solar_radiation": "solar_radiation_Quality_Code",
    "wind_gust": "wind_gust_Quality_Code",
    "snow_depth": "snow_depth_Quality_Code",
    "sea_level_pressure": "sea_level_pressure_Quality_Code",
    "sky_cover_1": "sky_cover_1_Quality_Code",
    "sky_cover_2": "sky_cover_2_Quality_Code",
    "sky_cover_3": "sky_cover_3_Quality_Code",
    "sky_cover_baseht_1": "sky_cover_baseht_1_Quality_Code",
    "sky_cover_baseht_2": "sky_cover_baseht_2_Quality_Code",
    "sky_cover_baseht_3": "sky_cover_baseht_3_Quality_Code",
    "precipitation_3h": "precipitation_3h_Quality_Code",
    "precipitation_6h": "precipitation_6h_Quality_Code",
    "precipitation_12h": "precipitation_12h_Quality_Code",
    "precipitation_24h": "precipitation_24h_Quality_Code",
    "precipitation_estimated": "precipitation_estimated_Quality_Code",
    "solar_radiation_estimated": "solar_radiation_estimated_Quality_Code",
    "temperature_max": "temperature_max_Quality_Code",
    "temperature_min": "temperature_min_Quality_Code",
    "dew_point_temperature_max": "dew_point_temperature_max_Quality_Code",
    "dew_point_temperature_min": "dew_point_temperature_min_Quality_Code",
    "relative_humidity_max": "relative_humidity_max_Quality_Code",
    "relative_humidity_min": "relative_humidity_min_Quality_Code",
    "wind_speed_max": "wind_speed_max_Quality_Code",
    "wind_speed_min": "wind_speed_min_Quality_Code",
    "station_level_pressure_max": "station_level_pressure_max_Quality_Code",
    "station_level_pressure_min": "station_level_pressure_min_Quality_Code",
    "sea_level_pressure_max": "sea_level_pressure_max_Quality_Code",
    "sea_level_pressure_min": "sea_level_pressure_min_Quality_Code",
}

# Variable descriptions and units
_VAR_DESCRIPTIONS = {
    "temperature": "Air temperature (°C)",
    "dew_point_temperature": "Dew point temperature (°C)",
    "wind_speed": "Wind speed (m/s)",
    "wind_direction": "Wind direction (degrees)",
    "precipitation": "Precipitation amount (mm)",
    "relative_humidity": "Relative humidity (%)",
    "station_level_pressure": "Station pressure (hPa)",
    "visibility": "Visibility (km)",
    "wet_bulb_temperature": "Wet bulb temperature (°C)",
    "solar_radiation": "Solar radiation (W/m²)",
    "wind_gust": "Wind gust speed (m/s)",
    "snow_depth": "Snow depth (mm)",
    "sea_level_pressure": "Sea level pressure (hPa)",
    "sky_cover_1": "Sky cover layer 1 (tenths)",
    "sky_cover_2": "Sky cover layer 2 (tenths)",
    "sky_cover_3": "Sky cover layer 3 (tenths)",
    "sky_cover_baseht_1": "Sky cover base height layer 1 (m)",
    "sky_cover_baseht_2": "Sky cover base height layer 2 (m)",
    "sky_cover_baseht_3": "Sky cover base height layer 3 (m)",
    "precipitation_3h": "3-hour precipitation (mm)",
    "precipitation_6h": "6-hour precipitation (mm)",
    "precipitation_12h": "12-hour precipitation (mm)",
    "precipitation_24h": "24-hour precipitation (mm)",
    "precipitation_estimated": "Estimated precipitation (mm)",
    "solar_radiation_estimated": "Estimated solar radiation (W/m²)",
    "temperature_max": "Maximum temperature (°C)",
    "temperature_min": "Minimum temperature (°C)",
    "dew_point_temperature_max": "Maximum dew point temperature (°C)",
    "dew_point_temperature_min": "Minimum dew point temperature (°C)",
    "relative_humidity_max": "Maximum relative humidity (%)",
    "relative_humidity_min": "Minimum relative humidity (%)",
    "wind_speed_max": "Maximum wind speed (m/s)",
    "wind_speed_min": "Minimum wind speed (m/s)",
    "station_level_pressure_max": "Maximum station pressure (hPa)",
    "station_level_pressure_min": "Minimum station pressure (hPa)",
    "sea_level_pressure_max": "Maximum sea level pressure (hPa)",
    "sea_level_pressure_min": "Minimum sea level pressure (hPa)",
}

# -----------------------------------------------------------------------------
# Globals & constants
# -----------------------------------------------------------------------------
_BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly"
_BY_YEAR = f"{_BASE_URL}/access/by-year"
_BY_STATION = f"{_BASE_URL}/access/by-station"

_CHUNK = 2 << 20  # 2 MiB streaming chunk size

# -----------------------------------------------------------------------------
# Station catalog helpers
# -----------------------------------------------------------------------------

def read_station_catalog(path: str | pathlib.Path) -> pd.DataFrame:
    """Return tidy **DataFrame** of the GHCNh station list indexed by *ID*."""
    df = pd.read_csv(path, dtype={"ID": str})
    return df.rename(columns=str.lower).set_index("id", drop=True)

# -----------------------------------------------------------------------------
# Download helpers
# -----------------------------------------------------------------------------

def _stream_to_file(url: str, dest: pathlib.Path) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    _logger.info("Downloading %s → %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
    return dest


def _build_url(station: str, *, year: Optional[int] = None, fmt: str = "psv") -> str:
    station = station.upper()
    if year is None:
        return f"{_BY_STATION}/psv/GHCNh_{station}_por.{fmt}"
    return f"{_BY_YEAR}/{year}/psv/GHCNh_{station}_{year}.{fmt}"


def download_station_file(
    station: str,
    *,
    year: Optional[int] = None,
    fmt: str = "psv",
    cache_dir: str | pathlib.Path = "./ghcnh_cache",
) -> pathlib.Path:
    """Download *station-year* or *period-of-record* file, returning **local path**."""
    url = _build_url(station, year=year, fmt=fmt)
    fname = pathlib.Path(cache_dir) / url.split("/")[-1]
    path = _stream_to_file(url, fname)
    return path


def open_ghcnh_dataframe(path: str | pathlib.Path) -> pd.DataFrame:
    """Load PSV or Parquet GHCNh file into a :class:`pandas.DataFrame`.

    The returned frame:

    * Has a **UTC `DatetimeIndex`**.
    * Leaves original column names intact (e.g. `temperature_Quality_Code`) so you
      can perform bespoke QC downstream.
    """
    path = pathlib.Path(path)
    df: pd.DataFrame
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="|", low_memory=False)

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
        df = df.set_index("DATE").rename_axis("time")
    else:  # POR file: build timestamp from Y/M/D/H/MIN columns
        dt = pd.to_datetime(
            df[["Year", "Month", "Day", "Hour", "Minute"]]
            .astype(int)
            .astype(str)
            .agg("-".join, axis=1),
            format="%Y-%m-%d-%h-%M",
            utc=True,
        )
        df = df.set_index(dt).rename_axis("time")
    return df


# ------------------------------------------------------------------
# Main dataset class
# ------------------------------------------------------------------

@dataclass
class StationDataset:
    """Enhanced wrapper around GHCNh data with pandas DataFrame as native data holder."""

    _df: pd.DataFrame
    _meta: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Populate _meta with station metadata if present in the DataFrame
        meta_fields = ["STATION", "Station_name", "LATITUDE", "LONGITUDE", "ELEVATION"]
        for field in meta_fields:
            if field not in self._meta and field in self._df.columns and len(self._df) > 0:
                self._meta[field] = self._df[field].iloc[0]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "StationDataset":
        """Create from a GHCNh file."""
        return cls.from_dataframe(open_ghcnh_dataframe(path))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "StationDataset":
        """Create from a pandas DataFrame."""
        # Ensure time is properly set as index
        if "time" in df.columns:
            df = df.set_index("time")
        
        return cls(df)

    # ------------------------------------------------------------------
    # Time handling utilities
    # ------------------------------------------------------------------

    
    def get_time_index(self) -> pd.DatetimeIndex:
        """Get the time index as a pandas DatetimeIndex."""
        return pd.DatetimeIndex(self._df.index)
    
    def get_time_range(self) -> Dict[str, Any]:
        """Get the time range of the dataset.
        
        Returns:
            Dictionary with start and end times
        """
        time_index = self.get_time_index()
        
        # Handle empty dataset
        if len(time_index) == 0:
            return {
                'start_time': None,
                'end_time': None,
                'duration_days': 0,
                'total_observations': 0
            }
        
        start_time = time_index.min()
        end_time = time_index.max()
        
        # Handle potential NaT values
        if start_time is pd.NaT or end_time is pd.NaT:
            return {
                'start_time': None,
                'end_time': None,
                'duration_days': 0,
                'total_observations': len(time_index)
            }
        
        # Ensure we have proper Timestamp objects
        if isinstance(start_time, pd.Timestamp) and isinstance(end_time, pd.Timestamp):
            duration = end_time - start_time
            duration_days = duration.days
        else:
            duration_days = 0
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration_days': duration_days,
            'total_observations': len(time_index)
        }

    def to_local_time(self, timezone: str) -> "StationDataset":
        """Convert UTC times to local timezone.
        
        Args:
            timezone: Timezone string (e.g., 'America/New_York')
            
        Returns:
            New StationDataset with local time coordinate
        """
        time_index = self.get_time_index()
        if time_index.tz is None:
            time_index = time_index.tz_localize('UTC')
        local_time = time_index.tz_convert(timezone)
        
        df = self._df.copy()
        df['local_time'] = local_time
        return StationDataset(df, _meta=self._meta.copy())

    def add_local_time(self) -> "StationDataset":
        """Automatically detect timezone and add local time coordinate."""
        if TimezoneFinder is None:
            _logger.warning("timezonefinder not installed → skipping local time coordinate")
            return self

        lon = float(self._df["LONGITUDE"].iloc[0]) if "LONGITUDE" in self._df.columns else np.nan
        lat = float(self._df["LATITUDE"].iloc[0]) if "LATITUDE" in self._df.columns else np.nan

        tzname = None
        try:
            tzname = TimezoneFinder().timezone_at(lng=lon, lat=lat)
        except Exception:
            pass

        if tzname is None:
            _logger.warning("Could not resolve timezone for station coords lat=%s lon=%s", lat, lon)
            return self

        return self.to_local_time(tzname)

    # ------------------------------------------------------------------
    # Cleaning / QC
    # ------------------------------------------------------------------
    def clean(self) -> "StationDataset":
        """Clean the dataset by handling missing values and QC."""
        df = self._df.copy()

        # replace sentinel missing values with NaN
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].replace(list(_MISSING_VALUES), np.nan)

        # trace precipitation -> 0
        if "precipitation_Measurement_Code" in df.columns:
            trace_mask = df["precipitation_Measurement_Code"].isin(list(_TRACE_CODES))
            if "precipitation" in df.columns:
                df.loc[trace_mask, "precipitation"] = 0.0
        # Ensure only DataFrame is passed
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        return StationDataset(df, _meta=self._meta.copy())

    def select_variables(self, variables: List[str]) -> "StationDataset":
        """Select specific variables and their associated metadata columns.
        
        Args:
            variables: List of variable names to select (e.g., ['temperature', 'precipitation'])
            
        Returns:
            New StationDataset with only the selected variables
        """
        if not variables:
            return self
            
        # Get all columns to keep
        columns_to_keep = []
        
        for var in variables:
            if var not in _AVAILABLE_VARS:
                _logger.warning(f"Variable '{var}' not found in available variables")
                continue
                
            # Add the main variable
            if var in self._df.columns:
                columns_to_keep.append(var)
            
            # Add associated metadata columns
            qc_col = _AVAILABLE_VARS[var]
            mc_col = f"{var}_Measurement_Code"
            rt_col = f"{var}_Report_Type"
            sc_col = f"{var}_Source_Code"
            ssi_col = f"{var}_Source_Station_ID"
            
            for col in [qc_col, mc_col, rt_col, sc_col, ssi_col]:
                if col in self._df.columns:
                    columns_to_keep.append(col)
        
        # Also keep station metadata
        for col in ['STATION', 'Station_name','LATITUDE', 'LONGITUDE', 'ELEVATION']:
            if col in self._df.columns:
                columns_to_keep.append(col)
        
        # Create new dataset with selected variables
        selected_df = self._df[columns_to_keep]
        # Ensure only DataFrame is passed
        if not isinstance(selected_df, pd.DataFrame):
            selected_df = pd.DataFrame(selected_df)
        return StationDataset(selected_df, _meta=self._meta.copy())

    # Convenience methods for common variable groups
    def get_temperature(self) -> "StationDataset":
        """Get temperature data with quality control."""
        return self.select_variables(['temperature'])

    def get_precipitation(self) -> "StationDataset":
        """Get precipitation data with quality control."""
        return self.select_variables(['precipitation'])

    def get_wind(self) -> "StationDataset":
        """Get wind speed and direction data with quality control."""
        return self.select_variables(['wind_speed', 'wind_direction'])

    def get_pressure(self) -> "StationDataset":
        """Get pressure data with quality control."""
        return self.select_variables(['station_level_pressure', 'sea_level_pressure'])

    def get_humidity(self) -> "StationDataset":
        """Get humidity-related data with quality control."""
        return self.select_variables(['relative_humidity', 'dew_point_temperature', 'wet_bulb_temperature'])

    def get_visibility(self) -> "StationDataset":
        """Get visibility data with quality control."""
        return self.select_variables(['visibility'])

    def get_clouds(self) -> "StationDataset":
        """Get cloud cover and height data with quality control."""
        return self.select_variables(['sky_cover_1', 'sky_cover_2', 'sky_cover_3', 
                                    'sky_cover_baseht_1', 'sky_cover_baseht_2', 'sky_cover_baseht_3'])

    def get_extended_precipitation(self) -> "StationDataset":
        """Get all precipitation variables (hourly, 3h, 6h, etc.) with quality control."""
        precip_vars = [var for var in _AVAILABLE_VARS.keys() if 'precipitation' in var]
        return self.select_variables(precip_vars)

    def get_data_availability(self) -> Dict[str, Dict[str, Any]]:
        """Get data availability statistics for all variables.
        
        Returns:
            Dictionary with availability stats for each variable
        """
        stats = {}
        
        for var in _AVAILABLE_VARS.keys():
            if var in self._df.columns:
                data = self._df[var]
                total_obs = len(data)
                valid_obs = data.notna().sum()
                missing_obs = total_obs - valid_obs
                
                stats[var] = {
                    'total_observations': total_obs,
                    'valid_observations': valid_obs,
                    'missing_observations': missing_obs,
                    'completeness_percent': (valid_obs / total_obs * 100) if total_obs > 0 else 0,
                    'description': _VAR_DESCRIPTIONS.get(var, 'No description available'),
                    'units': _VAR_DESCRIPTIONS.get(var, '').split('(')[-1].split(')')[0] if '(' in _VAR_DESCRIPTIONS.get(var, '') else ''
                }
        
        return stats

    def get_summary_statistics(self, variables: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for specified variables.
        
        Args:
            variables: List of variables to analyze. If None, analyze all available variables.
            
        Returns:
            Dictionary with statistics for each variable
        """
        if variables is None:
            variables = [var for var in _AVAILABLE_VARS.keys() if var in self._df.columns]
        
        stats = {}
        
        for var in variables:
            if var in self._df.columns:
                data = self._df[var].dropna()
                if len(data) > 0:
                    stats[var] = {
                        'count': int(len(data)),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'median': float(data.median()),
                        'q25': float(data.quantile(0.25)),
                        'q75': float(data.quantile(0.75))
                    }
                else:
                    stats[var] = {'count': 0, 'mean': np.nan, 'std': np.nan, 
                                'min': np.nan, 'max': np.nan, 'median': np.nan, 
                                'q25': np.nan, 'q75': np.nan}
        
        return stats

    def get_station_info(self) -> Dict[str, Any]:
        """Get station metadata information.
        
        Returns:
            Dictionary with station information
        """
        info = {}
        
        # Get coordinates if available
        if 'LATITUDE' in self._df.columns:
            info['latitude'] = float(self._df['LATITUDE'].iloc[0])
        if 'LONGITUDE' in self._df.columns:
            info['longitude'] = float(self._df['LONGITUDE'].iloc[0])
        if 'ELEVATION' in self._df.columns:
            info['elevation'] = float(self._df['ELEVATION'].iloc[0])
        
        # Get time range
        time_info = self.get_time_range()
        info.update(time_info)
        
        # Get data availability
        availability = self.get_data_availability()
        info['variables_available'] = list(availability.keys())
        info['total_variables'] = len(availability)
        
        return info

    def export_to_csv(self, filepath: str, variables: Optional[List[str]] = None) -> None:
        """Export selected variables to CSV file.
        
        Args:
            filepath: Path to output CSV file
            variables: List of variables to export. If None, export all variables.
        """
        if variables:
            ds = self.select_variables(variables)
        else:
            ds = self
        
        ds._df.to_csv(filepath)
        _logger.info(f"Exported data to {filepath}")

    def export_to_netcdf(self, filepath: str, variables: Optional[List[str]] = None) -> None:
        """Export selected variables to NetCDF file.
        
        Args:
            filepath: Path to output NetCDF file
            variables: List of variables to export. If None, export all variables.
        """
        if variables:
            ds = self.select_variables(variables)
        else:
            ds = self
        
        ds.to_xarray().to_netcdf(filepath)
        _logger.info(f"Exported data to {filepath}")

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate a comprehensive quality report for the dataset.
        
        Returns:
            Dictionary with quality assessment information
        """
        report = {
            'station_info': self.get_station_info(),
            'data_availability': self.get_data_availability(),
            'summary_statistics': self.get_summary_statistics(),
            'quality_issues': {}
        }
        
        # Check for potential quality issues
        for var, qc_col in _AVAILABLE_VARS.items():
            if var in self._df.columns and qc_col in self._df.columns:
                qc_data = self._df[qc_col]
                total_qc = len(qc_data)
                
                if total_qc > 0:
                    # Count different QC codes
                    qc_counts = qc_data.value_counts().to_dict()
                    
                    # Identify potential issues
                    issues = []
                    for code, count in qc_counts.items():
                        if code not in _GOOD_QC_CODES["source_code_set_1"]:
                            issues.append(f"QC code '{code}': {count} observations")
                    
                    if issues:
                        report['quality_issues'][var] = issues
        
        return report

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(
        self, 
        variable: str = "temperature", 
        freq: str = "1h", 
        method: str = "mean"
        ) -> "StationDataset":
        """Resample to *freq* using variable-aware aggregation rules."""
        df = self._df.copy()[variable]
        
        # Create resampler
        resampler = df.resample(freq)
        
        # Perform aggregation
        aggregated_df = resampler.agg(method)
        # Ensure only DataFrame is passed
        if not isinstance(aggregated_df, pd.DataFrame):
            aggregated_df = pd.DataFrame(aggregated_df)
        return StationDataset(aggregated_df, _meta=self._meta.copy())

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------
    def to_xarray(self) -> xr.Dataset:
        """Convert to xarray Dataset."""
        return xr.Dataset.from_dataframe(self._df)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with DatetimeIndex."""
        return self._df.copy()

    def to_pandas_series(self, variable: str) -> pd.Series:
        """Convert a single variable to pandas Series with DatetimeIndex.
        
        Args:
            variable: Name of the variable to convert
            
        Returns:
            pandas Series with DatetimeIndex
        """
        if variable not in self._df.columns:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        
        # Ensure only Series is returned
        result = self._df[variable]
        if not isinstance(result, pd.Series):
            result = pd.Series(result)
        return result

    def qc_and_split_by_source_station(
        self, variable: str, out_dir: str = "ghcnh_cache/cleaned", strict: bool = False
    ):
        """
        For a given variable, drop rows with NaN Source_Station_ID, split by unique source station,
        apply QC, and return a dict mapping source_station_id to a new StationDataset for each cleaned sub-DataFrame.
        Ensures DATE and station metadata columns are included. Source_Code is saved as stringified int.
        """
        src_col = f"{variable}_Source_Station_ID"
        qc_col = f"{variable}_Quality_Code"
        code_col = f"{variable}_Source_Code"
        # Metadata columns to always keep if present
        meta_cols = [
            "time", "STATION", "Station_name", "LATITUDE", "LONGITUDE"
        ]
        if src_col not in self._df.columns or qc_col not in self._df.columns:
            raise ValueError(f"Required columns for {variable} not found in dataset")

        # 1. Drop rows with NaN Source_Station_ID
        df = self._df.dropna(subset=[src_col]).copy()

        result = {}
        # 2. Split by unique source station
        for source_station in df[src_col].unique():
            sub_df = df[df[src_col] == source_station].copy()
            good_qc = _GOOD_QC_CODES["source_code_set_1"]
            # Set bad QC values to NaN for the variable
            mask = ~sub_df[qc_col].isin(good_qc)
            sub_df.loc[mask, variable] = np.nan
            # Optionally, drop rows where variable is now NaN
            sub_df = sub_df.dropna(subset=[variable])
            # Ensure DATE and station metadata columns are present if available
            keep_cols = meta_cols + [
                c for c in [
                    variable,
                    f"{variable}_Measurement_Code",
                    f"{variable}_Quality_Code",
                    f"{variable}_Report_Type",
                    f"{variable}_Source_Code",
                    f"{variable}_Source_Station_ID",
                ] if c in sub_df.columns
            ]
            # Ensure Source_Code is saved as stringified int (no .0)
            if code_col in sub_df.columns:
                sub_df.loc[:, code_col] = (
                    sub_df.loc[:, code_col]
                    .astype(float)
                    .astype(int)
                    .astype(str)
                )
            # Only keep the columns in keep_cols if they exist, but preserve all columns in the DataFrame
            # (for downstream flexibility)
            # Return as a new StationDataset
            result[source_station] = StationDataset(sub_df, _meta=self._meta.copy())
        return result

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        time_range = self.get_time_range()
        return (f"<StationDataset n={len(self._df)} "
                f"({time_range['start_time']} to {time_range['end_time']})>\n"
                f"{self._df.head()}") 
