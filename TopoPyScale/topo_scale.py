"""
Toposcale functionalities
S. Filhol, Oct 2021

======= Organization of input data to Toposcale ======
dem_description (nb_points)
    X
    Y,
    elev,
    slope,
    aspect,
    svf,
    x_ERA,
    y_ERA,
    elev_ERA,

horizon_angles (nb_points * bins)
solar_geom (nb_points * time * 2)
surface_climate_data (spatial + time + var)
plevels_climate_data (spatial + time + var + plevel)

======= Dataset naming convention:  =========
ds_surf => dataset direct from ERA5 single level surface
ds_plev => dataset direct from ERA5 pressure levels

ds_surf_pt => 3*3 grid ERA5 single surface level around a given point (lat,lon)
ds_plev_pt => 3*3 grid ERA5 pressure level around a given point (lat,lon)

plev_interp => horizontal interpolation of the 3*3 grid at the point (lat,lon) for the pressure levels
surf_interp => horizontal interpolation of the 3*3 grid at the point (lat,lon) for the single surface level

top => closest pressure level above the point (lat,lon,elev) for each timesep
bot => closest pressure level below the point (lat,lon,elev) for each timesep

down_pt => downscaled data time series (t, u, v, q, LW, SW, tp)
"""
import pdb
import traceback

import pandas as pd
import xarray as xr
from exceptiongroup import print_exc
from pyproj import Transformer
import numpy as np
import sys, time
import datetime as dt
from TopoPyScale import meteo_util as mu
from TopoPyScale import topo_utils as tu

import os, glob
from pathlib import Path
from typing import Union


# Physical constants
g = 9.81  # Acceleration of gravity [ms^-1]
R = 287.05  # Gas constant for dry air [JK^-1kg^-1]



def clear_files(path: Union[Path, str]):
    if not isinstance(path, Path):
        path = Path(path)

    files = [f for f in path.rglob('*') if f.is_file()]
    for file in files:
        file.unlink()
    print(f'{path} cleaned')

def pt_downscale_interp(row, ds_plev_pt, ds_surf_pt, meta):
    # If "r" (Relative Humidity) not available, calculate it
    if "r" not in ds_plev_pt.variables:
        calculate_relative_humidity(ds_plev_pt, ds_surf_pt)

    pt_id = row.point_name
    print(f'Downscaling t,q,p,tp,ws, wd for point: {pt_id}')

    # ====== Horizontal interpolation ====================
    interp_method = meta.get('interp_method')

    # convert gridcells coordinates from WGS84 to DEM projection
    lons, lats = np.meshgrid(ds_plev_pt.longitude.values, ds_plev_pt.latitude.values)
    trans = Transformer.from_crs("epsg:4326", "epsg:" + str(meta.get('target_epsg')), always_xy=True)
    Xs, Ys = trans.transform(lons.flatten(), lats.flatten())
    Xs = Xs.reshape(lons.shape)
    Ys = Ys.reshape(lons.shape)

    dist = np.sqrt((row.x - Xs) ** 2 + (row.y - Ys) ** 2)
    if interp_method == 'idw':
        idw = 1 / (dist ** 2)
        weights = idw / np.sum(idw)  # normalize idw to sum(idw) = 1
    elif interp_method == 'linear':
        weights = dist / np.sum(dist)
    else:
        sys.exit('ERROR: interpolation method not available')

    # create a dataArray of weights to then propagate through the dataset
    da_idw = xr.DataArray(data=weights,
                              coords={
                                  "latitude": ds_plev_pt.latitude.values,
                                  "longitude": ds_plev_pt.longitude.values,
                              },
                              dims=["latitude", "longitude"]
                              )
    # convert to float64 consequence of CDS-BETA change to dtype in netcdf download
    ds_plev_pt = ds_plev_pt.astype('float64')
    da_idw = da_idw.astype('float64')
    ds_surf_pt = ds_surf_pt.astype('float64')

    dw = xr.Dataset.weighted(ds_plev_pt, da_idw)
    plev_interp = dw.sum(['longitude', 'latitude'],
                             keep_attrs=True)  # compute horizontal inverse weighted horizontal interpolation
    dww = xr.Dataset.weighted(ds_surf_pt, da_idw)
    surf_interp = dww.sum(['longitude', 'latitude'],
                              keep_attrs=True)  # compute horizontal inverse weighted horizontal interpolation

    # ========= Converting z from [m**2 s**-2] to [m] asl =======
    # plev_interp.z.values = plev_interp.z.values / g
    # plev_interp.z.attrs = {'units': 'm', 'standard_name': 'Elevation', 'Long_name': 'Elevation of plevel'}

    # surf_interp.z.values = surf_interp.z.values / g  # convert geopotential height to elevation (in m), normalizing by g
    # surf_interp.z.attrs = {'units': 'm', 'standard_name': 'Elevation', 'Long_name': 'Elevation of ERA5 surface'}

    # ============ Extract specific humidity (q) for both dataset ============
    surf_interp = mu.dewT_2_q_magnus(surf_interp, mu.var_era_surf)
    plev_interp = mu.t_rh_2_dewT(plev_interp, mu.var_era_plevel)

    if 'valid_time' in plev_interp.coords:
        if 'time' in plev_interp.dims:
            plev_interp = plev_interp.isel(time=0, drop=True)
        plev_interp = plev_interp.rename({"valid_time": "time"})

    if 'pressure_level' in plev_interp.coords and 'level' not in plev_interp.coords:
        plev_interp = plev_interp.rename({"pressure_level": "level"})

    down_pt = xr.Dataset(coords={
            'time': plev_interp.time,
            #'point_name': pd.to_numeric(pt_id, errors='ignore')
            'point_name': pt_id
    })


    if (row.elevation < plev_interp.z.isel(level=-1)).sum():
        pt_elev_diff = np.round(np.min(row.elevation - plev_interp.z.isel(level=-1).values), 0)
        print(f"---> WARNING: Point {pt_id} is {pt_elev_diff} m lower than the {plev_interp.isel(level=-1).level.data} hPa geopotential\n=> "
                  "Values sampled from Psurf and lowest Plevel. No vertical interpolation")

        diff_z = (plev_interp.where(plev_interp.z > row.elevation).z - row.elevation)
        try:
            ind_z_top = diff_z.argmin("level")
        except Exception:
            traceback.print_exc()
            ind_z_top = 0
        try:
            top = plev_interp.isel(level=ind_z_top)  # Assume level "0"
            down_pt['t'] = top['t']
            down_pt['u'] = top.u
            down_pt['v'] = top.v
            down_pt['q'] = top.q
            down_pt['p'] = top.level * (10 ** 2) * np.exp(
                    -(row.elevation - top.z) / (0.5 * (top.t + down_pt.t) * R / g))  # Pressure in bar
        except Exception:
            traceback.print_exc()

    else:
        # ========== Vertical interpolation at the DEM surface z  ===============
        ind_z_bot = (plev_interp.where(plev_interp.z < row.elevation).z - row.elevation).argmax('level')

        # In case upper pressure level elevation is not high enough, stop process and print error
        try:
            ind_z_top = (plev_interp.where(plev_interp.z > row.elevation).z - row.elevation).argmin('level')
        except:
            raise ValueError(
                f'ERROR: Upper pressure level {plev_interp.level.min().values} hPa geopotential is lower than cluster mean elevation {row.elevation} {plev_interp.z}')
               

        top = plev_interp.isel(level=ind_z_top)
        bot = plev_interp.isel(level=ind_z_bot)

        # Preparing interpolation weights for linear interpolation =======================
        dist = np.array([np.abs(bot.z - row.elevation).values, np.abs(top.z - row.elevation).values])
        # idw = 1/dist**2
        # weights = idw / np.sum(idw, axis=0)
        weights = dist / np.sum(dist, axis=0)

        # ============ Creating a dataset containing the downscaled timeseries ========================
        down_pt['t'] = bot.t * weights[1] + top.t * weights[0]
        down_pt['u'] = bot.u * weights[1] + top.u * weights[0]
        down_pt['v'] = bot.v * weights[1] + top.v * weights[0]
        down_pt['q'] = bot.q * weights[1] + top.q * weights[0]
        down_pt['p'] = top.level * (10 ** 2) * np.exp(
                -(row.elevation - top.z) / (0.5 * (top.t + down_pt.t) * R / g))  # Pressure in bar

        # ======= logic  to compute ws, wd without loading data in memory, and maintaining the power of dask
    down_pt['month'] = ('time', down_pt.time.dt.month.data)
    print("precip lapse rate = " + str(meta.get('precip_lapse_rate_flag')))
    if meta.get('precip_lapse_rate_flag'):
        monthly_coeffs = xr.Dataset(
                {
                    'coef': (['month'], [0.35, 0.35, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35])
                },
                coords={'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
            )
        down_pt['precip_lapse_rate'] = (1 + monthly_coeffs.coef.sel(month=down_pt.month.values).data * (
                    row.elevation - surf_interp.z) * 1e-3) / \
                                           (1 - monthly_coeffs.coef.sel(month=down_pt.month.values).data * (
                                                   row.elevation - surf_interp.z) * 1e-3)
    else:
        down_pt['precip_lapse_rate'] = down_pt.t * 0 + 1

    down_pt['tp'] = down_pt.precip_lapse_rate * surf_interp.tp * 1 / meta.get('tstep') * 10 ** 3  # Convert to mm/hr
    down_pt['theta'] = np.arctan2(-down_pt.u, -down_pt.v)
    down_pt['theta_neg'] = (down_pt.theta < 0) * (down_pt.theta + 2 * np.pi)
    down_pt['theta_pos'] = (down_pt.theta >= 0) * down_pt.theta
    down_pt = down_pt.drop('theta')
    down_pt['wd'] = (down_pt.theta_pos + down_pt.theta_neg)  # direction in Rad
    down_pt['ws'] = np.sqrt(down_pt.u ** 2 + down_pt.v ** 2)
    down_pt = down_pt.drop(['theta_pos', 'theta_neg', 'month'])

    down_pt.t.attrs = {'units': 'K', 'long_name': 'Temperature', 'standard_name': 'air_temperature'}
    down_pt.q.attrs = {'units': 'kg kg**-1', 'long_name': 'Specific humidity', 'standard_name': 'specific_humidity'}
    down_pt.u.attrs = {'units': 'm s**-1', 'long_name': 'U component of wind', 'standard_name': 'eastward_wind'}
    down_pt.v.attrs = {'units': 'm s**-1', 'long_name': 'V component of wind', 'standard_name': 'northward wind'}
    down_pt.p.attrs = {'units': 'bar', 'long_name': 'Pression atmospheric', 'standard_name': 'pression_atmospheric'}

    down_pt.ws.attrs = {'units': 'm s**-1', 'long_name': 'Wind speed', 'standard_name': 'wind_speed'}
    down_pt.wd.attrs = {'units': 'deg', 'long_name': 'Wind direction', 'standard_name': 'wind_direction'}
    down_pt.tp.attrs = {'units': 'mm hr**-1', 'long_name': 'Precipitation', 'standard_name': 'precipitation'}
    down_pt.precip_lapse_rate.attrs = {'units': 'mm hr**-1',
                                           'long_name': 'Precipitation after lapse-rate correction',
                                           'standard_name': 'precipitation_after_lapse-rate_correction'}

    down_pt.attrs = {'title':'Downscale point using TopoPyScale',
                          'date_created':dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
    output_directory = meta.get('output_directory')
    out_file = output_directory / f'down_pt_{pt_id}.nc'
    print(f'---> Storing point {pt_id} to {out_file}')

    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in down_pt.data_vars}
    down_pt.to_netcdf(out_file, engine='h5netcdf', encoding=encoding)
    down_pt.close()

    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in surf_interp.data_vars}
    output_dir = meta.get('output_dir')
    surf_interp.to_netcdf(output_dir / f'surf_interp_{pt_id}.nc', engine='h5netcdf', encoding=encoding)
    down_pt = None
    surf_interp = None
    top, bot = None, None


def pt_downscale_radiations(row, ds_solar, horizon_da, meta):
    # insrt here downscaling routine for sw and lw
    # save file final file
    pt_id = row.point_name

    file_pattern = meta.get('file_pattern')
    output_directory = meta.get('output_directory')
    output_dir = meta.get('output_dir')
    print(f'Downscaling LW, SW for point: {pt_id}')

    down_pt = xr.open_dataset(output_directory / f'down_pt_{pt_id}.nc', engine='h5netcdf')
    _ = down_pt.load()
    down_pt.close()
    down_pt = _

    surf_interp = xr.open_dataset(output_dir / f'surf_interp_{pt_id}.nc', engine='h5netcdf')

    # ======== Longwave downward radiation ===============
    x1, x2 = 0.43, 5.7
    sbc = 5.67e-8

    down_pt = mu.mixing_ratio(down_pt, mu.var_era_plevel)
    down_pt = mu.vapor_pressure(down_pt, mu.var_era_plevel)
    surf_interp = mu.mixing_ratio(surf_interp, mu.var_era_surf)
    surf_interp = mu.vapor_pressure(surf_interp, mu.var_era_surf)

    # ========= Compute clear sky emissivity ===============
    down_pt['cse'] = 0.23 + x1 * (down_pt.vp / down_pt.t) ** (1 / x2)
    surf_interp['cse'] = 0.23 + x1 * (surf_interp.vp / surf_interp.t2m) ** (1 / x2)
    # Calculate the "cloud" emissivity, UNIT OF STRD (Jjoel_testJun2023/m2)
    surf_interp['cle'] = (surf_interp.strd / pd.Timedelta('1H').seconds) / (sbc * surf_interp.t2m ** 4) - \
                         surf_interp['cse']
    # Use the former cloud emissivity to compute the all sky emissivity at subgrid.
    surf_interp['aef'] = down_pt['cse'] + surf_interp['cle']
    if meta.get('lw_terrain_flag'):
        down_pt['LW'] = row.svf * surf_interp['aef'] * sbc * down_pt.t ** 4 + \
                        0.5 * (1 + np.cos(row.slope)) * (1 - row.svf) * 0.99 * 5.67e-8 * (273.15 ** 4)
    else:
        down_pt['LW'] = row.svf * surf_interp['aef'] * sbc * down_pt.t ** 4

    kt = surf_interp.ssrd * 0
    ds_solar = ds_solar.persist()
    sunset = ds_solar.sunset.astype(bool).compute()
    mu0 = ds_solar.mu0
    SWtoa = ds_solar.SWtoa

    # pdb.set_trace()
    kt[~sunset] = (surf_interp.ssrd[~sunset] / pd.Timedelta('1H').seconds) / SWtoa[~sunset]  # clearness index
    kt[kt < 0] = 0
    kt[kt > 1] = 1
    kd = 0.952 - 1.041 * np.exp(-1 * np.exp(2.3 - 4.702 * kt))  # Diffuse index

    surf_interp['SW'] = surf_interp.ssrd / pd.Timedelta('1H').seconds
    surf_interp['SW'][surf_interp['SW'] < 0] = 0
    surf_interp['SW_diffuse'] = kd * surf_interp.SW
    down_pt['SW_diffuse'] = row.svf * surf_interp.SW_diffuse

    surf_interp['SW_direct'] = surf_interp.SW - surf_interp.SW_diffuse
    # scale direct solar radiation using Beer's law (see Aalstad 2019, Appendix A)
    ka = surf_interp.ssrd * 0
    # pdb.set_trace()
    ka[~sunset] = (g * mu0[~sunset] / down_pt.p) * np.log(SWtoa[~sunset] / surf_interp.SW_direct[~sunset])
    # Illumination angle
    down_pt['cos_illumination_tmp'] = mu0 * np.cos(row.slope) + np.sin(ds_solar.zenith) * \
                                      np.sin(row.slope) * np.cos(ds_solar.azimuth - row.aspect)
    down_pt['cos_illumination'] = down_pt.cos_illumination_tmp * (
            down_pt.cos_illumination_tmp > 0)  # remove selfdowing ccuring when |Solar.azi - aspect| > 90
    down_pt = down_pt.drop(['cos_illumination_tmp'])
    illumination_mask = down_pt['cos_illumination'] < 0
    illumination_mask = illumination_mask.compute()
    down_pt['cos_illumination'][illumination_mask] = 0

    # Binary shadow masks.
    horizon = horizon_da.sel(x=row.x, y=row.y, azimuth=np.rad2deg(ds_solar.azimuth), method='nearest')
    shade = (horizon > ds_solar.elevation)
    down_pt['SW_direct_tmp'] = down_pt.t * 0
    down_pt['SW_direct_tmp'][~sunset] = SWtoa[~sunset] * np.exp(
        -ka[~sunset] * down_pt.p[~sunset] / (g * mu0[~sunset]))
    down_pt['SW_direct'] = down_pt.t * 0
    down_pt['SW_direct'][~sunset] = down_pt.SW_direct_tmp[~sunset] * (
            down_pt.cos_illumination[~sunset] / mu0[~sunset]) * (1 - shade)
    down_pt['SW'] = down_pt.SW_diffuse + down_pt.SW_direct

    # currently drop azimuth and level as they are coords. Could be passed to variables instead.
    # round(5) required to sufficiently represent specific humidty, q (eg typical value 0.00078)
    down_pt = down_pt.drop(['level']).round(5)

    # adding metadata
    down_pt.LW.attrs = {'units': 'W m**-2', 'long_name': 'Surface longwave radiation downwards',
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        'standard_name': 'longwave_radiation_downward'}
    down_pt.cse.attrs = {'units': 'xxx', 'standard_name': 'Clear sky emissivity'}
    down_pt = down_pt.drop(['SW_direct_tmp'])
    down_pt.SW.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar radiation downwards',
                        'standard_name': 'shortwave_radiation_downward'}
    down_pt.SW_diffuse.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar diffuse radiation downwards',
                                'standard_name': 'shortwave_diffuse_radiation_downward'}
    ver_dict = tu.get_versionning()
    down_pt.attrs = {'title': 'Downscaled timeseries with TopoPyScale',
                     'created with': 'TopoPyScale, see more at https://topopyscale.readthedocs.io',
                      'package_version':ver_dict.get('package_version'),
                      'git_commit': ver_dict.get('git_commit'),
                     'url_TopoPyScale': 'https://github.com/ArcticSnow/TopoPyScale',
                     'date_created': dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in down_pt.data_vars}
    output_path = output_directory / file_pattern.replace('*', str(pt_id))
    print(f"---> Storing point {pt_id} to {output_path}")
    down_pt.to_netcdf(output_path, engine='h5netcdf', encoding=encoding, mode='a')

    # Clear memory
    down_pt, surf_interp = None, None
    ds_solar = None
    kt, ka, kd, sunset = None, None, None, None
    sunset = None
    horizon = None
    shade = None


def calculate_relative_humidity(pressure_levels_ds, surface_ds):
    """
    Calculates relative humidity and adds it to the pressure levels dataset.

    Reads pressure from the 'level' coordinate (assumed hPa/mb) and converts to Pa.
    Reads temperature ('t') and specific humidity ('q') from data variables.

    Assumes the input datasets contain:
    - pressure_levels_ds:
        - 'level': Coordinate with pressure levels (assumed in hPa/mb) - Used for pressure
        - 't': Temperature at pressure levels (in Kelvin, K) - Used in calculation
        - 'q': Specific humidity at pressure levels (dimensionless, kg/kg) - Used in calculation
    - surface_ds:
        - 'sp': Surface pressure (in Pascals, Pa) - Checked for existence

    Adds the following variable to the pressure_levels_ds:
    - 'r': Relative humidity (fraction, 0-1)

    Parameters
    ----------
    pressure_levels_ds : xarray.Dataset
        Dataset containing temperature ('t'), specific humidity ('q'),
        and a 'level' coordinate representing pressure in hPa/mb.
    surface_ds : xarray.Dataset
        Dataset containing surface variables, including surface pressure ('sp').

    Returns
    -------
    xarray.Dataset
        The pressure_levels_ds dataset modified to include relative humidity ('r').
    """
    # Constants
    epsilon = 0.621981  # Ratio of molar masses of water vapor to dry air (approx 18.015 / 28.9644)

    # --- Input Validation ---
    required_plevel_vars = ['t', 'q'] # Check for 't' and 'q' variables
    missing_plevel_vars = [var for var in required_plevel_vars if var not in pressure_levels_ds]
    if missing_plevel_vars:
        raise ValueError(f"Pressure levels dataset is missing required variables: {missing_plevel_vars}")

    if 'level' not in pressure_levels_ds.coords:
        raise ValueError("Pressure levels dataset is missing the 'level' coordinate.")

    required_surface_vars = ['sp']
    missing_surface_vars = [var for var in required_surface_vars if var not in surface_ds]
    if missing_surface_vars:
        raise ValueError(f"Surface dataset is missing required variables: {missing_surface_vars}")

    # --- Extract Data ---
    # Get pressure from the 'level' coordinate and convert hPa/mb -> Pa
    # Ensure the pressure coordinate is aligned for broadcasting with t and q
    try:
        pressure_pa = pressure_levels_ds['level'] * 100.0
        pressure_pa.attrs['units'] = 'Pa' # Add units attribute for clarity
        # Xarray will automatically broadcast the 1D pressure_pa coordinate
        # during calculations with multi-dimensional t and q if dimensions match.
    except KeyError:
        raise ValueError("Could not find 'level' coordinate in pressure_levels_ds.")
    except Exception as e:
        raise RuntimeError(f"Error processing 'level' coordinate: {e}")


    # --- Perform Calculations Directly ---
    q_safe = xr.where(pressure_levels_ds['q'] >= 1.0, 1.0 - 1e-12, pressure_levels_ds['q'])
    mixing_ratio = q_safe / (1.0 - q_safe)

    # 2. Calculate saturation vapor pressure (e_s)
    # Formula: e_s = 0.61094 * exp(17.625 * T_c / (T_c + 243.04)) (in kPa)
    temp_celsius = pressure_levels_ds['t'] - 273.15
    svp_pa = (0.61094 * np.exp(17.625 * temp_celsius / (temp_celsius + 243.04))) * 1000

    # 3. Calculate saturation mixing ratio (w_s) using pressure in Pa
    # Formula: w_s = epsilon * e_s / (p - e_s)
    safe_svp = xr.where(svp_pa >= pressure_pa, pressure_pa - 1e-6, svp_pa)
    w_s = (epsilon * safe_svp) / (pressure_pa - safe_svp)

    # 4. Calculate relative humidity (RH) from mixing ratio (w) and saturation mixing ratio (w_s)
    # Employs WMO Eq. 4.A.16: RH = (w / (epsilon + w)) * ((epsilon + w_s) / w_s)
    rh_num = mixing_ratio * (epsilon + w_s)
    rh_den = w_s * (epsilon + mixing_ratio)

    # 4. Calculate relative humidity (RH) from mixing ratio (w) and saturation mixing ratio (w_s)
    # Employs WMO Eq. 4.A.16: RH = (w / (epsilon + w)) * ((epsilon + w_s) / w_s)
    rh_num = mixing_ratio * (epsilon + w_s)
    rh_den = w_s * (epsilon + mixing_ratio)

    # relative_humidity = np.divide(rh_num, rh_den, out=np.full_like(rh_num, np.nan), where=rh_den!=0)
    relative_humidity = xr.where(rh_den != 0, rh_num / rh_den, np.nan)

    # Handle edge cases and clip
    relative_humidity = xr.where((w_s <= 1e-12) & (mixing_ratio > 1e-12), 1.0, relative_humidity)
    relative_humidity = xr.where((w_s <= 1e-12) & (mixing_ratio <= 1e-12), 0.0, relative_humidity)
    relative_humidity = relative_humidity.clip(min=0.0, max=1.0)

    # --- Add Result to Pressure Levels Dataset ---
    # Ensure the output dimensions match the input t and q variables
    relative_humidity = relative_humidity.rename('r') # Rename DataArray before assigning
    relative_humidity.attrs['long_name'] = 'Relative Humidity'
    relative_humidity.attrs['units'] = '1' # Dimensionless fraction
    relative_humidity.attrs['standard_name'] = 'relative_humidity'
    relative_humidity.attrs['calculation'] = 'Calculated from level(Pa), t(K), q(kg/kg) using WMO Eq. 4.A.16'

    # Assign the calculated array back to the original pressure levels dataset
    pressure_levels_ds['r'] = relative_humidity

    print("Relative humidity ('r') calculated and added to the pressure levels dataset.")

    return pressure_levels_ds


def downscale_climate(project_directory,
                      climate_directory,
                      downscaling_tmp_dir,
                      output_directory,
                      df_centroids,
                      horizon_da,
                      ds_solar,
                      target_EPSG,
                      start_date,
                      end_date,
                      tstep,
                      interp_method='idw',
                      lw_terrain_flag=True,
                      precip_lapse_rate_flag=False,
                      file_pattern='down_pt*.nc',
                      n_core=4):
    """
    Function to perform downscaling of climate variables (t,q,u,v,tp,SW,LW) based on Toposcale logic

    Args:
        project_directory (str): path to project root directory
        climate_directory (str):
        output_directory (path):
        df_centroids (dataframe): containing a list of point for which to downscale (includes all terrain data)
        horizon_da (dataarray): horizon angles for a list of azimuth
        target_EPSG (int): EPSG code of the DEM
        start_date:
        end_date:
        interp_method (str): interpolation method for horizontal interp. 'idw' or 'linear'
        lw_terrain_flag (bool): flag to compute contribution of surrounding terrain to LW or ignore
        tstep (str): timestep of the input data, default = 1H
        file_pattern (str): filename pattern for storing downscaled points, default = 'down_pt_*.nc'
        n_core (int): number of core on which to distribute processing

    Returns:
        dataset: downscaled data organized with time, point_name, lat, long
    """
    global pt_downscale_interp
    global pt_downscale_radiations
    global _subset_climate_dataset

    print('\n-------------------------------------------------')
    print('           TopoScale - Downscaling\n')
    print('---> Downscaling climate to list of points using TopoScale')
    clear_files(output_directory)

    start_time = time.time()
    tstep_dict = {'1H': 1, '3H': 3, '6H': 6}
    n_digits = len(str(df_centroids.index.max()))

    # =========== Open dataset with Dask =================
    tvec = pd.date_range(start_date, pd.to_datetime(end_date) + pd.to_timedelta('1D'), freq=tstep, inclusive='left')

    flist_PLEV = glob.glob(f'{climate_directory}/PLEV*.nc')
    flist_SURF = glob.glob(f'{climate_directory}/SURF*.nc')

    if 'object' in list(df_centroids.dtypes):
        pass
    else:
        df_centroids['dummy'] = 'nn'

    def _open_dataset_climate(flist, required_vars, time_coord_name='time'):
        process_start_time = pd.Timestamp(start_date).to_datetime64()
        process_end_time = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).to_datetime64()

        # Filter files to include only those containing all required variables
        filtered_files = []
        skipped_files = []

        for f in flist:
            try:
                with xr.open_dataset(f) as ds:
                    file_min_time = ds[time_coord_name].min().values
                    file_max_time = ds[time_coord_name].max().values
                    has_some_to_process_time = file_min_time < process_end_time and file_max_time >= process_start_time
                    if has_some_to_process_time:
                        # Check if all required variables are in this file
                        all_variables_found = True
                        for var_spec in required_vars:
                            # Check if at least one allowed name for this concept exists
                            found_this_concept = any(allowed_name in ds.variables for allowed_name in var_spec)
                            if not found_this_concept:
                                all_variables_found = False
                        if all_variables_found:
                            filtered_files.append(f)
                        else:
                            skipped_files.append(f)
                    # else:
                    #     print(f"INFO: File {f} does not cover the requested time period: {process_start_time} to {process_end_time}")
            except Exception as e:
                print(f"ERROR: Failed to open file {f}: {e}")
                skipped_files.append(f)
                # traceback.print_exc()

        # Print information about filtered files
        if skipped_files:
            print(f"WARNING: Skipped {len(skipped_files)} files missing required variables.")
            print(f"Required variables: {required_vars}")
            print(f"Files skipped: {[os.path.basename(f) for f in skipped_files]}")

        if not filtered_files:
            raise ValueError("No files contain all required variables!")

        # Open only the filtered files
        ds_ = xr.open_mfdataset(filtered_files, parallel=False, concat_dim="time", combine='nested', coords='minimal')
        # Modify longitudes to -180 to 180
        if ds_.longitude.values.min() >= 0:
            ds_["longitude"] = np.mod(ds_.longitude.values + 180.0, 360.0) - 180.0
            # ds_ = ds_.assign_coords(longitude=new_longitude_values)

        # Rename variables to the first element of each sublist in required_vars
        print("Applying variable renaming rules...")
        rename_map = {}
        current_vars_in_ds = set(ds_.variables)  # Get variable names from the combined dataset

        for var_spec in required_vars:
            if not var_spec:  # Skip empty specs
                continue
            desired_name = var_spec[0]
            aliases = var_spec[1:]

            for alias in aliases:
                if alias in current_vars_in_ds and alias != desired_name:
                    rename_map[alias] = desired_name

        if rename_map:
            ds_ = ds_.rename(rename_map)

        return ds_

    def _subset_climate_dataset(ds_, row, type):
        print('Preparing {} for point {}'.format(type, row.point_name))
        # =========== Extract the 3*3 cells centered on a given point ============
        ind_lat = np.abs(ds_.latitude - row.y).argmin()
        if ds_.latitude[0] > ds_.latitude[-1]:
            ind_lat -= 1
        ind_lon = np.abs(ds_.longitude - row.x).argmin()

        try:
            ds_tmp = ds_.isel(latitude= [ind_lat - 1, ind_lat, ind_lat + 1],
                              longitude=[ind_lon - 1, ind_lon, ind_lon + 1]).copy()
        except Exception as e:
            print(f"ERROR: Point {row.point_name} is outside the domain of the dataset.")
            print(f"Point coordinates: ({row.y}, {row.x})")
            print(f"Dataset boundaries (lats and lots): ({ds_.latitude.min().values}, {ds_.latitude.max().values}, {ds_.longitude.min().values}, {ds_.longitude.max().values})")
            raise e

        # convert geopotential height to elevation (in m), normalizing by g
        ds_tmp['z'] = ds_tmp.z / g

        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds_tmp.data_vars}
        ds_tmp.to_netcdf(output_directory / f'ds_{type}_pt_{row.point_name}.nc', engine='h5netcdf',
                         encoding=encoding)
        ds_ = None
        ds_tmp = None

    # OPEN PRESSURE LEVELS META DATASET
    import dask
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds_ = _open_dataset_climate(flist_PLEV, [['z', 'Z'], ['t', 'T'], ['u', 'U'], ['v', 'V'], ['q', 'Q']])

        if 'valid_time' in ds_.coords:
            if 'time' in ds_.dims:
                ds_ = ds_.isel(time=0, drop=True)
            ds_ = ds_.rename({"valid_time": "time"})

        ds_plev = ds_.sel(time=slice(tvec.values[0], tvec.values[-1]))  # tvec.values)
        if "R" in ds_plev.variables and "r" not in ds_plev.variables:
            ds_plev = ds_plev.rename({"R": "r"})

    row_list = []
    ds_list = []
    for _, row in df_centroids.iterrows():
        row_list.append(row)
        ds_list.append(ds_plev)

    fun_param = zip(ds_list, row_list, ['plev'] * len(row_list))  # construct here the tuple that goes into the pooling for arguments
    tu.multithread_pooling(_subset_climate_dataset, fun_param, n_threads=n_core)

    # OPEN SURFACE VARIABLES META DATASET
    ds_ = _open_dataset_climate(flist_SURF, [['z', 'Z'], ['d2m', '2D'], ['strd', 'STRD'], ['ssrd', 'SSRD'], ['sp', 'SP'], ['tp', 'TP'], ['t2m', '2T']])
    if 'valid_time' in ds_.coords:
        if 'time' in ds_.dims:
            ds_ = ds_.isel(time=0, drop=True)
        ds_ = ds_.rename({"valid_time": "time"})

    ds_surf = ds_.sel(time=slice(tvec.values[0], tvec.values[-1]))  # tvec.values)
    # ds_surf = ds_surf.sel(time=tvec.values)

    ds_list = []
    for _, _ in df_centroids.iterrows():
        ds_list.append(ds_surf)

    fun_param = zip(ds_list, row_list, ['surf'] * len(row_list))  # construct here the tuple that goes into the pooling for arguments
    tu.multithread_pooling(_subset_climate_dataset, fun_param, n_threads=n_core)
    fun_param = None
    ds_surf = None

    # Preparing list to feed into Pooling
    surf_pt_list = []
    plev_pt_list = []
    ds_solar_list = []
    horizon_da_list = []
    row_list = []
    meta_list = []
    i = 0
    for _, row in df_centroids.iterrows():
        surf_pt_list.append(xr.open_dataset(output_directory / f'ds_surf_pt_{row.point_name}.nc', engine='h5netcdf'))
        plev_pt_list.append(xr.open_dataset(output_directory / f'ds_plev_pt_{row.point_name}.nc', engine='h5netcdf'))
        ds_solar_list.append(ds_solar.sel(point_name=row.point_name))
        horizon_da_list.append(horizon_da)
        row_list.append(row)
        meta_list.append({'interp_method': interp_method,
                          'lw_terrain_flag': lw_terrain_flag,
                          'tstep': tstep_dict.get(tstep),
                          'n_digits': n_digits,
                          'file_pattern': file_pattern,
                          'target_epsg':target_EPSG,
                          'precip_lapse_rate_flag':precip_lapse_rate_flag,
                          'output_directory':downscaling_tmp_dir,
                          'output_dir': output_directory,
                          'lw_terrain_flag':lw_terrain_flag})
        i+=1

    fun_param = zip(row_list, plev_pt_list, surf_pt_list, meta_list)  # construct here the tuple that goes into the pooling for arguments
    if n_core == 1:
        for row, ds_plev_pt, ds_surf_pt, meta in fun_param:
            pt_downscale_interp(row, ds_plev_pt, ds_surf_pt, meta)
    else:
        tu.multicore_pooling(pt_downscale_interp, fun_param, n_core)
    fun_param = None
    plev_pt_list = None
    surf_pt_list = None

    fun_param = zip(row_list, ds_solar_list, horizon_da_list, meta_list)  # construct here tuple to feed pool function's argument
    if n_core == 1:
        for row, ds_solar, horizon_da, meta in fun_param:
            pt_downscale_radiations(row, ds_solar, horizon_da, meta)
    else:
        tu.multicore_pooling(pt_downscale_radiations, fun_param, n_core)
    fun_param = None
    ds_solar_list = None
    horizon_da_list = None

    # print timer to console
    print('---> Downscaling finished in {}s'.format(np.round(time.time() - start_time), 1))


def read_downscaled(path='outputs/down_pt*.nc'):
    """
    Function to read downscaled points saved into netcdf files into one single dataset using Dask

    Args:
        path (str): path and pattern to use into xr.open_mfdataset

    Returns:
        dataset: merged dataset readily to use and loaded in chuncks via Dask
    """

    down_pts = xr.open_mfdataset(path, concat_dim='point_name', combine='nested', parallel=False)
    return down_pts
