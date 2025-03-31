#!/usr/bin/env python
"""
Retrieve ERA5 data from Google Cloud Platform (GCP) in ZARR format. CLI also.

This script provides a command-line interface for downloading and processing ERA5 climate data
stored in ZARR format on Google Cloud Platform, using the TopoPyScale configuration file and
optional command-line parameters.

Example usage:
    python fetch_era5_zarr.py --config_file=topopyscale_config.yml --output_dir=/path/to/output
    python fetch_era5_zarr.py --start_date=2020-01-01 --end_date=2020-12-31 --lat_north=60 --lat_south=40 --lon_east=20 --lon_west=-10 --output_dir=/path/to/output
"""
import os
import sys
import time
import pandas as pd
import xarray as xr
import numpy as np
import dask
import fire
import yaml
from munch import Munch
from typing import List, Optional, Dict, Any, Union
import dotenv
from datetime import datetime


def is_valid_netcdf(file_path):
    """Check if a NetCDF file is valid."""
    try:
        xr.open_dataset(file_path).close()
        return True
    except:
        return False


def hypsometric_pressure(surface_pressure, geopotential_height, temperature):
    """
    Calculate the pressure at a given geopotential height using the hypsometric equation.

    Args:
        surface_pressure (array): Surface pressure in Pascals.
        geopotential_height (array): Geopotential height in meters.
        temperature (array): Temperature in Kelvin.

    Returns:
        array: Pressure at the given geopotential height in Pascals.
    """
    L = 0.0065  # Temperature lapse rate (K/m)
    R = 287.05  # Gas constant for dry air (J/(kg·K))
    g = 9.81  # Gravitational acceleration (m/s²)
    M = 0.0289644  # Molar mass of Earth's air (kg/mol)

    pressure = surface_pressure * (1 - (L * geopotential_height) / (R * temperature)) ** (g * M / (R * L))
    return pressure


def q_2_rh(temp, surface_pressure, geopotential_height, qair):
    """
    Convert specific humidity (q) to relative humidity (RH).

    Args:
        temp (array): Temperature in Kelvin.
        surface_pressure (array): Surface pressure in Pascals.
        geopotential_height (array): Geopotential height in meters.
        qair (array): Specific humidity in kg/kg.

    Returns:
        array: Relative humidity in percentage (0-100%).
    """
    pressure = hypsometric_pressure(surface_pressure, geopotential_height, temp)
    mr = qair / (1 - qair)
    e = mr * pressure / (0.62197 + mr)
    es = 611.2 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))
    rh = (e / es) * 100
    rh = np.clip(rh, 0, 100)  # Ensure RH is within bounds
    return rh


def process_month(ds, start, end, lons, lats, levels, output_dir, surface_vars, pressure_vars, s2l_surf, s2l_pl):
    """Process and save data for a given month."""
    # Define output file paths
    surf_dir = output_dir  # os.path.join(output_dir, 'surf')
    plev_dir = output_dir  # os.path.join(output_dir, 'plev')
    os.makedirs(surf_dir, exist_ok=True)
    os.makedirs(plev_dir, exist_ok=True)

    surface_file = os.path.join(surf_dir, f"SURF_{start.strftime('%Y%m')}.nc")
    pressure_file = os.path.join(plev_dir, f"PLEV_{start.strftime('%Y%m')}.nc")

    if is_valid_netcdf(surface_file) and is_valid_netcdf(pressure_file):
        print(f"Skipping {start.strftime('%Y%m')} - Files already exist and are valid.")
        return

    d_params = {}
    if lons is not None:
        d_params['longitude'] = lons
    if lats is not None:
        d_params['latitude'] = lats
    if levels is not None:
        d_params['level'] = levels

    # Process and save surface fields
    temp_surface_file = surface_file + ".tmp"
    if not is_valid_netcdf(surface_file):
        start_time = time.time()
        print(f"Writing {temp_surface_file}...")
        dask.compute(
            ds.sel(time=slice(start, end), **d_params)[[s2l_surf[s] for s in surface_vars]].to_netcdf(temp_surface_file,
                                                                                                      mode='w',
                                                                                                      format="NETCDF4",
                                                                                                      engine="netcdf4"),
            scheduler='processes')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time {surface_file}: {elapsed_time:.6f} seconds")
        ds_surface = xr.open_dataset(temp_surface_file)
        ds_surface = ds_surface.rename({s2l_surf[s]: s for s in surface_vars})
        ds_surface.to_netcdf(temp_surface_file + ".remap", mode='w')
        os.rename(temp_surface_file + ".remap", surface_file)
        os.remove(temp_surface_file)
    else:
        print(f"Opening existing {surface_file}")
        ds_surface = xr.open_dataset(surface_file)

    # Process and save pressure-level fields
    if not is_valid_netcdf(pressure_file):
        sp = ds_surface['sp'].values.copy()
        ds_surface.close()

        temp_pressure_file = pressure_file + ".tmp"
        print(f"Writing {temp_pressure_file}...")
        start_time = time.time()
        dask.compute(
            ds.sel(time=slice(start, end), **d_params)[[s2l_pl[s] for s in pressure_vars]].to_netcdf(temp_pressure_file,
                                                                                                     mode='w',
                                                                                                     format="NETCDF4",
                                                                                                     engine="netcdf4"),
            scheduler='processes')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time {pressure_file}: {elapsed_time:.6f} seconds")

        ds_pressure = xr.open_dataset(temp_pressure_file)
        ds_pressure = ds_pressure.rename({s2l_pl[s]: s for s in pressure_vars})
        q = ds_pressure['q'].values
        t = ds_pressure['t'].values
        z = ds_pressure['z'].values
        sp = np.expand_dims(sp, axis=1).repeat(q.shape[1], axis=1)
        rh = q_2_rh(t, sp, z, q)
        ds_pressure['r'] = xr.DataArray(rh, dims=ds_pressure['q'].dims, coords=ds_pressure['q'].coords)
        ds_pressure['r'].attrs['long_name'] = 'Relative Humidity'
        ds_pressure['r'].attrs['units'] = '%'
        ds_pressure.to_netcdf(temp_pressure_file + ".remap", mode='w')
        ds_pressure.close()
        os.rename(temp_pressure_file + ".remap", pressure_file)
        os.remove(temp_pressure_file)


def read_config(config_file: str) -> Munch:
    """
    Read a YAML configuration file and return a Munch object.

    Args:
        config_file: Path to the YAML configuration file

    Returns:
        Munch object with configuration data

    Raises:
        FileNotFoundError: If the config file does not exist
        yaml.YAMLError: If the YAML file is invalid
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        return Munch.fromDict(config)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def parse_comma_list(value: Union[str, List, None]) -> Optional[List[str]]:
    """
    Parse a comma-separated string into a list.

    Args:
        value: Input string with comma-separated values, or a list

    Returns:
        List of strings, or None if input is empty
    """
    if not value:
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.split(',')]
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    return None


def retrieve_era5_zarr(
        start_date: str,
        end_date: str,
        output_dir: str,
        lat_north: float,
        lat_south: float,
        lon_east: float,
        lon_west: float,
        dataset_path: str,
        surf_vars: Optional[List[str]] = None,
        plev_vars: Optional[List[str]] = None,
        plevels: Optional[List[str]] = None,
        num_threads: int = 1
):
    """
    Retrieves ERA5 data from GCP in ZARR format and processes it into NetCDF files.

    Args:
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        output_dir: Directory to write output files
        lat_north: North latitude of bounding box
        lat_south: South latitude of bounding box
        lon_east: East longitude of bounding box
        lon_west: West longitude of bounding box
        dataset_path: Path to ZARR dataset (gs://, s3://, etc.)
        surf_vars: List of surface variables to download
        plev_vars: List of pressure level variables to download
        plevels: List of pressure levels to download
        num_threads: Number of threads for parallel processing

    Returns:
        Dictionary with paths to the output files
    """
    print(f"\n---> Loading ERA5 data from ZARR at {dataset_path}")

    # Set default variables if not specified
    if surf_vars is None:
        surf_vars = ['d2m', 't2m', 'sp', 'ssrd', 'strd', 'tp', 'z']

    if plev_vars is None:
        plev_vars = ['q', 't', 'u', 'v', 'z']

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    surf_dir = output_dir  # os.path.join(output_dir, 'surf')
    plev_dir = output_dir  # os.path.join(output_dir, 'plev')
    os.makedirs(surf_dir, exist_ok=True)
    os.makedirs(plev_dir, exist_ok=True)

    # Load environment variables for cloud authentication
    dotenv.load_dotenv()

    print(f"Opening ZARR dataset from {dataset_path}")

    # Convert longitude range to 0-360 if needed
    lons = np.arange(lon_west % 360, lon_east % 360, 0.25)
    lats = np.arange(lat_south, lat_north, 0.25)

    # Open ZARR dataset
    ds = xr.open_zarr(dataset_path, chunks=None)

    # Map short names to long names
    s2l_surf = {}  # Short to Long dict for surface variables
    s2l_pl = {}  # Short to Long dict for pressure level variables
    drop_vars = []

    for d in ds.data_vars:
        if hasattr(ds[d], 'short_name'):
            if ds[d].short_name not in surf_vars + plev_vars:
                drop_vars.append(d)
            elif "level" in ds[d].coords:
                s2l_pl[ds[d].short_name] = d
            else:
                s2l_surf[ds[d].short_name] = d

    ds.close()

    # Reopen with optimized chunking and dropping unnecessary variables
    ds = xr.open_zarr(dataset_path, drop_variables=drop_vars, chunks='auto')

    # Determine pressure levels to keep
    def get_sublist_until_value(arr, target_value):
        idx = np.where(arr == target_value)[0]
        if len(idx) == 0:
            raise ValueError(f"Target value {target_value} not found in the array.")
        start_idx = idx[-1]  # Last occurrence of target_value
        return arr[start_idx:]

    # Define the pressure levels to keep (from the target value up)
    # Default to 700 hPa and above
    target_level = 700
    if plevels and len(plevels) > 0:
        # Convert string levels to integers
        int_levels = [int(p) for p in plevels]
        target_level = max(int_levels)  # Use the highest level as target

    levels = get_sublist_until_value(ds.level.values, target_level)

    # Create time ranges for processing
    time_ranges = pd.date_range(start=start_date, end=end_date, freq='MS')
    total_months = len(time_ranges)

    # Process each month
    start_time = time.time()

    # Create dataframe to track files
    df = pd.DataFrame()
    df['time_start'] = time_ranges
    df['time_end'] = df.time_start.apply(
        lambda x: (x + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

    # Set output file paths
    df['surf_file'] = df.time_start.apply(lambda x: os.path.join(surf_dir, f"SURF_{x.strftime('%Y%m')}.nc"))
    df['plev_file'] = df.time_start.apply(lambda x: os.path.join(plev_dir, f"PLEV_{x.strftime('%Y%m')}.nc"))

    # Check for existing files
    df['surf_exists'] = df.surf_file.apply(lambda x: is_valid_netcdf(x))
    df['plev_exists'] = df.plev_file.apply(lambda x: is_valid_netcdf(x))

    # Report on existing files
    if df.surf_exists.sum() > 0:
        print("\nSurface data files already found:")
        print(df.loc[df.surf_exists, 'surf_file'].apply(lambda x: os.path.basename(x)).tolist())

    if df.plev_exists.sum() > 0:
        print("\nPressure level data files already found:")
        print(df.loc[df.plev_exists, 'plev_file'].apply(lambda x: os.path.basename(x)).tolist())

    # Process months that need downloading
    for idx, row in df.iterrows():
        if not (row.surf_exists and row.plev_exists):
            month_start_time = time.time()
            print(f"\nProcessing month {idx + 1}/{total_months}: {row.time_start.strftime('%Y-%m')}")

            process_month(
                ds=ds,
                start=row.time_start,
                end=row.time_end,
                lons=lons,
                lats=lats,
                levels=levels,
                output_dir=output_dir,
                surface_vars=surf_vars,
                pressure_vars=plev_vars,
                s2l_surf=s2l_surf,
                s2l_pl=s2l_pl
            )

            month_end_time = time.time()
            elapsed_time = month_end_time - month_start_time

            completed_months = idx + 1
            avg_time_per_month = (month_end_time - start_time) / completed_months
            remaining_months = total_months - completed_months
            estimated_remaining_time = avg_time_per_month * remaining_months

            print(f"Processed {row.time_start.strftime('%Y-%m')} in {elapsed_time:.2f} seconds.")
            print(
                f"Completed {completed_months}/{total_months} months. Estimated time remaining: {estimated_remaining_time / 60:.2f} minutes.")

    total_time = time.time() - start_time
    print(f"Script completed in {total_time / 60:.2f} minutes.")

    # Return paths to output files
    return {
        'surface_files': df.surf_file.tolist(),
        'plevel_files': df.plev_file.tolist()
    }


def main(
        config_file: str,
        start_date: str = None,
        end_date: str = None,
        lat_north: float = None,
        lat_south: float = None,
        lon_east: float = None,
        lon_west: float = None,
        output_dir: str = None,
        surf_vars: str = None,
        plev_vars: str = None,
        plevels: str = None,
        num_threads: int = 1,
        verbose: bool = False
) -> Dict[str, Any]:
    """
    Download and process ERA5 data from GCP in ZARR format.

    This function reads configuration from a TopoPyScale YAML file and/or command-line arguments,
    and processes ERA5 data from ZARR format to NetCDF files.

    Args:
        config_file: Path to TopoPyScale configuration file (YAML)
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        lat_north: North latitude of bounding box
        lat_south: South latitude of bounding box
        lon_east: East longitude of bounding box
        lon_west: West longitude of bounding box
        output_dir: Directory to save the processed files
        surf_vars: Surface variables to process (comma-separated)
        plev_vars: Pressure level variables to process (comma-separated)
        plevels: Pressure levels to process (comma-separated)
        num_threads: Number of threads for parallel processing
        verbose: Enable verbose output

    Returns:
        Dictionary containing information about processed files

    Raises:
        ValueError: If required parameters are missing
    """
    # Initialize parameters
    params = {
        'start_date': start_date,
        'end_date': end_date,
        'lat_north': lat_north,
        'lat_south': lat_south,
        'lon_east': lon_east,
        'lon_west': lon_west,
        'output_dir': output_dir,
        'dataset_path': None,
        'surf_vars': None,
        'plev_vars': None,
        'plevels': None,
        'num_threads': num_threads
    }

    # Read configuration from YAML file
    try:
        config = read_config(config_file)

        # Extract parameters from config
        if 'project' in config:
            if 'start' in config.project:
                params['start_date'] = params['start_date'] or config.project.start
            if 'end' in config.project:
                params['end_date'] = params['end_date'] or config.project.end
            params['output_dir'] = os.path.join(config.project.directory, config.climate.era5.path)

            if 'extent' in config.project:
                # In config: [lat_north, lat_south, lon_east, lon_west]
                extent = config.project.extent
                if isinstance(extent, list) and len(extent) == 4:
                    params['lat_north'] = params['lat_north'] or float(extent[0])
                    params['lat_south'] = params['lat_south'] or float(extent[1])
                    params['lon_east'] = params['lon_east'] or float(extent[2])
                    params['lon_west'] = params['lon_west'] or float(extent[3])

        if 'climate' in config and 'era5' in config.climate:
            if 'plevels' in config.climate.era5:
                params['plevels'] = params['plevels'] or config.climate.era5.plevels
            if 'dataset_path' in config.climate.era5:
                params['dataset_path'] = params['dataset_path'] or config.climate.era5.dataset_path
            if 'num_threads' in config.climate.era5:
                params['num_threads'] = params['num_threads'] or int(config.climate.era5.num_threads)


        if verbose:
            print(f"Loaded configuration from {config_file}")

    except (FileNotFoundError, yaml.YAMLError) as e:
        sys.stderr.write(f"Error reading configuration file: {e}\n")
        sys.exit(1)

    # Process list parameters
    params['surf_vars'] = parse_comma_list(surf_vars)
    params['plev_vars'] = parse_comma_list(plev_vars)
    params['plevels'] = parse_comma_list(plevels)

    # Set default dataset path if not provided
    if not params['dataset_path']:
        params['dataset_path'] = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

    if verbose:
        print("Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

    # Call the function with the parameters
    try:
        files = retrieve_era5_zarr(
            start_date=params['start_date'],
            end_date=params['end_date'],
            output_dir=params['output_dir'],
            lat_north=params['lat_north'],
            lat_south=params['lat_south'],
            lon_east=params['lon_east'],
            lon_west=params['lon_west'],
            dataset_path=params['dataset_path'],
            surf_vars=params['surf_vars'],
            plev_vars=params['plev_vars'],
            plevels=params['plevels'],
            num_threads=params['num_threads']
        )

        if verbose:
            print("Processing completed successfully.")

        return files
    except Exception as e:
        sys.stderr.write(f"Error during processing: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)