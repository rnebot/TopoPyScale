import xarray as xr
import pandas as pd
import numpy as np
import os
import time
import dotenv

"""
ERA5 Monthly Data Processor

This module processes ERA5 data stored in Zarr format by splitting it into monthly NetCDF files. 
It handles both surface and pressure-level fields, calculates derived fields (e.g., relative humidity), 
and organizes the output files by month.

Features:
- Validates existing files to avoid redundant processing.
- Calculates relative humidity using the hypsometric equation.
- Processes data efficiently in chunks based on monthly intervals.

Functions:
- is_valid_netcdf: Checks the validity of a NetCDF file.
- hypsometric_pressure: Computes pressure at a given height using the hypsometric equation.
- q_2_rh: Converts specific humidity to relative humidity.
- process_month: Processes and saves data for a specific month.
- main: The main entry point for processing data, which accepts parameters for start and end dates, 
  output directory, and dataset path.

Usage:
- Run the script directly for default processing or import as a module and call `main()` with custom parameters.
"""
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
    
    pressure = surface_pressure * (1 - (L * geopotential_height) / (R * temperature))**(g * M / (R * L))
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
    surface_file = os.path.join(output_dir, f"SURF_{start.strftime('%Y%m')}_{end.strftime('%Y%m')}.nc")
    pressure_file = os.path.join(output_dir, f"PLEV_{start.strftime('%Y%m')}_{end.strftime('%Y%m')}.nc")

    if is_valid_netcdf(surface_file) and is_valid_netcdf(pressure_file):
        print(f"Skipping {start.strftime('%Y-%m')} - Files already exist and are valid.")
        return

    d_params = {}
    if lons is not None:
        d_params['longitude'] = lons
    if lats is not None:
        d_params['latitude'] = lats
    if levels is not None:
        d_params['level'] = levels
    ds_month = ds.sel(time=slice(start, end), **d_params)

    # Process and save surface fields
    temp_surface_file = surface_file + ".tmp"
    if not is_valid_netcdf(surface_file):
        ds_surface = ds_month[[s2l_surf[s] for s in surface_vars]]
        start_time = time.time()
        ds_surface.to_netcdf(temp_surface_file, mode='w')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time {surface_file}: {elapsed_time:.6f} seconds")
        ds_surface = xr.open_dataset(temp_surface_file)
        ds_surface = ds_surface.rename({s2l_surf[s]: s for s in surface_vars})
        ds_surface.to_netcdf(temp_surface_file+".remap", mode='w')
        os.rename(temp_surface_file+".remap", surface_file)
        os.remove(temp_surface_file)
    else:
        ds_surface = xr.open_dataset(surface_file)
        # ds_surface = ds_surface.rename({s2l_surf[s]: s for s in surface_vars})
        # ds_surface.to_netcdf(temp_surface_file, mode='w')
        # os.rename(temp_surface_file, surface_file)

    # Process and save pressure-level fields
    if not is_valid_netcdf(pressure_file):
        temp_pressure_file = pressure_file + ".tmp"
        ds_pressure = ds_month[[s2l_pl[s] for s in pressure_vars]]
        start_time = time.time()
        ds_pressure.to_netcdf(temp_pressure_file, mode='w')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time {pressure_file}: {elapsed_time:.6f} seconds")

        ds_pressure = xr.open_dataset(temp_pressure_file)
        ds_pressure = ds_pressure.rename({s2l_pl[s]: s for s in pressure_vars})
        ds_pressure = ds_pressure.rename({s2l_pl[s]: s for s in pressure_vars})
        sp = ds_surface['sp'].values
        q = ds_pressure['q'].values
        t = ds_pressure['t'].values
        z = ds_pressure['z'].values
        rh = q_2_rh(t, sp, z, q)
        ds_pressure['r'] = xr.DataArray(rh, dims=ds_pressure['q'].dims, coords=ds_pressure['q'].coords)
        ds_pressure['r'].attrs['long_name'] = 'Relative Humidity'
        ds_pressure['r'].attrs['units'] = '%'
        ds_pressure.to_netcdf(temp_pressure_file+".remap", mode='w')
        os.rename(temp_pressure_file+".remap", pressure_file)
        os.remove(temp_pressure_file)



def main(start_date, end_date, lons, lats, output_dir, dataset_path):
    """Main function to process ERA5 data."""
    print(f"Downloading ERA5 from {start_date} to {end_date}...")
    dotenv.load_dotenv()
    os.makedirs(output_dir, exist_ok=True)

    ds = xr.open_zarr(dataset_path, chunks=None, storage_options=dict(token='anon'))

    surface_vars = ['d2m', 't2m', 'sp', 'ssrd', 'strd', 'tp', 'z']
    pressure_vars = ['q', 't', 'u', 'v', 'z']

    s2l_surf = {}  # Shor to Long dict
    s2l_pl = {}
    for d in ds.data_vars:
        if "level" in ds[d].coords:
            s2l_pl[ds[d].short_name] = d
        else:
            s2l_surf[ds[d].short_name] = d

    def get_sublist_until_value(arr, target_value):
        # Find the last occurrence of the target value
        idx = np.where(arr == target_value)[0]

        if len(idx) == 0:
            raise ValueError("Target value not found in the array.")

        # Get the starting index (last occurrence of target_value)
        start_idx = idx[-1]  # Last occurrence of target_value

        # Return sublist from the last index to the start_idx
        return arr[start_idx:]

    levels = get_sublist_until_value(ds.level.values, 700)

    time_ranges = pd.date_range(start=start_date, end=end_date, freq='MS')
    total_months = len(time_ranges)

    start_time = time.time()
    for idx, start in enumerate(time_ranges, start=1):
        month_start_time = time.time()
        end = (start + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        print(f"Processing month {idx}/{total_months}")
        process_month(ds, start, end, lons, lats, levels, output_dir, surface_vars, pressure_vars, s2l_surf, s2l_pl)
        month_end_time = time.time()

        elapsed_time = month_end_time - month_start_time
        completed_months = idx
        avg_time_per_month = (month_end_time - start_time) / completed_months
        remaining_months = total_months - completed_months
        estimated_remaining_time = avg_time_per_month * remaining_months

        print(f"Processed {start.strftime('%Y-%m')} in {elapsed_time:.2f} seconds.")
        print(f"Completed {completed_months}/{total_months} months. Estimated time remaining: {estimated_remaining_time / 60:.2f} minutes.")

    total_time = time.time() - start_time
    return f"Script completed in {total_time / 60:.2f} minutes."


if __name__ == "__main__":
    # Example usage
    result = main(
        start_date="1985-01-01",
        end_date="2015-12-31",
        lons=np.arange(-18.5%360, -13.25%360, 0.25),
        lats=np.arange(27.5, 29.5, 0.25),
        output_dir="/home/rnebot/GoogleDrive/AA_GENESIS/Datos/ERA5_ZARR/",
        dataset_path="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        # dataset_path="s3://spi-pamir-c7-sdsc/era5_data/central_asia.zarr/"
    )

    print(result)
