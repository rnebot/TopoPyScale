"""
Set of functions to work with DEMs
S. Filhol, Oct 2021

TODO:
- an improvement could be to first compute horizons, and then SVF to avoid computing horizon twice
"""

import sys
import rasterio
from pyproj import Transformer
import pandas as pd
import numpy as np
import xarray as xr
from topocalc import gradient
from topocalc import viewf
from topocalc import horizon
import time
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc
import os, shutil
from TopoPyScale import topo_export as te
import rioxarray
from pathlib import Path

def convert_epsg_pts(xs,ys, epsg_src=4326, epsg_tgt=3844):
    """
    Simple function to convert a list of points from one projection to another oen using PyProj

    Args:
        xs (array): 1D array with X-coordinate expressed in the source EPSG
        ys (array): 1D array with Y-coordinate expressed in the source EPSG
        epsg_src (int): source projection EPSG code
        epsg_tgt (int): target projection EPSG code

    Returns: 
        array: Xs 1D arrays of the point coordinates expressed in the target projection
        array: Ys 1D arrays of the point coordinates expressed in the target projection
    """
    print('Convert coordinates from EPSG:{} to EPSG:{}'.format(epsg_src, epsg_tgt))
    trans = Transformer.from_crs("epsg:{}".format(epsg_src), "epsg:{}".format(epsg_tgt), always_xy=True)
    Xs, Ys = trans.transform(xs, ys)
    return Xs, Ys


def get_extent_latlon(dem_file, epsg_src):
    """
    Function to extract DEM extent in Lat/Lon

    Args:
        dem_file (str): path to DEM file (GeoTiFF)
        epsg_src (int): EPSG projection code

    Returns: 
        dict: extent in lat/lon, {latN, latS, lonW, lonE}
    """
    with rasterio.open(dem_file) as rf:
        xs, ys = [rf.bounds.left, rf.bounds.right], [rf.bounds.bottom, rf.bounds.top]
    trans = Transformer.from_crs("epsg:{}".format(epsg_src), "epsg:4326", always_xy=True)
    lons, lats = trans.transform(xs, ys)
    extent = {'latN': lats[1],
              'latS': lats[0],
              'lonW': lons[0],
              'lonE': lons[1]}
    return extent


def extract_pts_param(df_pts, ds_param, method='nearest'):
    """
    Function to sample DEM parameters for a list point. This is used as an alternative the the TopoSub method, to perform downscaling at selected locations (x,y)
    WARNING: the projection and coordiante system of the EDM and point coordinates MUST be the same!

    Args:
        df_pts (dataframe): list of points coordinates with coordinates in (x,y).
        ds_param (dataset): dem parameters
        method (str): sampling method. Supported 'nearest', 'linear' interpolation, 'idw' interpolation (inverse-distance weighted)

    Returns: 
        dataframe: df_pts updated with new columns ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']
    """
    print('\n---> Extracting DEM parameters for the given list of point coordinates')
    # delete columns in case they already exist
    df_pts = df_pts.drop(['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf'], errors='ignore')
    # create columns, filled with 0
    df_pts[['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']] = 0

    if method == 'nearest':
        for i, row in df_pts.iterrows():
            d_mini = ds_param.sel(x=row.x, y=row.y, method='nearest')
            df_pts.loc[i, ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']] = np.array((d_mini.elevation.values,
                                                                                                   d_mini.slope.values,
                                                                                                   d_mini.aspect.values,
                                                                                                   d_mini.aspect_cos,
                                                                                                   d_mini.aspect_sin,
                                                                                                   d_mini.svf.values))
    elif method == 'idw' or method == 'linear':
        for i, row in df_pts.iterrows():
            ind_lat = np.abs(ds_param.y-row.y).argmin()
            ind_lon = np.abs(ds_param.x-row.x).argmin()
            ds_param_pt = ds_param.isel(y=[ind_lat-1, ind_lat, ind_lat+1], x=[ind_lon-1, ind_lon, ind_lon+1])
            Xs, Ys = np.meshgrid(ds_param_pt.x.values, ds_param_pt.y.values)

            if method == 'idw':
                dist = np.sqrt((row.x - Xs)**2 + (row.y - Ys)**2)
                idw = 1/(dist**2)
                weights = idw / np.sum(idw)  # normalize idw to sum(idw) = 1

            if method == 'linear':
                dist = np.sqrt((row.x - Xs)**2 + (row.y - Ys)**2)
                weights = dist / np.sum(dist)
            da_idw = xr.DataArray(data=weights,
                                  coords={
                                    "y": ds_param_pt.y.values,
                                    "x": ds_param_pt.x.values,
                              },
                              dims=["y", "x"]
                              )
            dw = xr.Dataset.weighted(ds_param_pt, da_idw)
            d_mini = dw.sum(['x', 'y'], keep_attrs=True)
            df_pts.loc[i, ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']] = np.array((d_mini.elevation.values,
                                                                                                   d_mini.slope.values,
                                                                                                   d_mini.aspect.values,
                                                                                                   d_mini.aspect_cos,
                                                                                                   d_mini.aspect_sin,
                                                                                                 d_mini.svf.values))
    else:
        raise ValueError('ERROR: Method not implemented. Only nearest, linear or idw available')
    return df_pts

def compute_dem_param(dem_file, fname='ds_param.nc', project_directory=Path('./'), output_folder='outputs', extent=None):
    """
    Function to compute and derive DEM parameters: slope, aspect, sky view factor

    Args:
        dem_file (str): path to raster file (geotif). Raster must be in local cartesian coordinate system (e.g. UTM)

    Returns:  
        dataset: x, y, elev, slope, aspect, svf

    """
    pdir = project_directory
    file_ds = pdir / output_folder / fname
    if file_ds.is_file():
        print(f'\n---> Dataset {fname} found.')
        ds = xr.open_dataset(file_ds)

    else:
        if Path(dem_file).is_file():
            print(f'\n---> No {fname} Dataset found. DEM {dem_file} available.')
            ds = rioxarray.open_rasterio(dem_file).to_dataset('band')
            ds = ds.rename({1: 'elevation'})

        else:
            raise ValueError(f'ERROR: No DEM or dataset available')

    # Cut the DEM to the extent of the study area
    if extent is not None:
        print('\n---> Cropping DEM to the extent of the study area')
        y1 = extent['latS'] if ds.y[0] < ds.y[-1] else extent['latN']
        y2 = extent['latN'] if ds.y[0] < ds.y[-1] else extent['latS']
        x1 = extent['lonW'] if ds.x[0] < ds.x[-1] else extent['lonE']
        x2 = extent['lonE'] if ds.x[0] < ds.x[-1] else extent['lonW']
        epsilon = 1e-5
        if y1 < y2:
            y1 += epsilon; y2 -= epsilon
        else:
            y1 -= epsilon; y2 += epsilon
        if x1 < x2:
            x1 += epsilon; x2 -= epsilon
        else:
            x1 -= epsilon; x2 += epsilon
        ds = ds.sel(y=slice(y1, y2), x=slice(x1, x2))

    var_in = list(ds.variables.keys())
    print('\n---> Extracting DEM parameters (slope, aspect, svf)')
    any_change = ('slope' not in var_in) or ('aspect' not in var_in) or ('svf' not in var_in)
    if any_change:
        dx = ds.x.diff('x').median().values
        dy = ds.y.diff('y').median().values
        dem_arr = ds.elevation.values
        dem_arr = xr.where(dem_arr < 0, 0, dem_arr)
        ds.elevation.values = dem_arr
        if ('slope' not in var_in) or ('aspect' not in var_in):
            print('Computing slope and aspect ...')
            slope, aspect = gradient.gradient_d8(dem_arr, dx, dy)
            ds['slope'] = (["y", "x"], slope)
            ds['aspect'] = (["y", "x"], np.deg2rad(aspect))
            if 'aspect_cos' not in var_in:
                ds['aspect_cos'] = (["y", "x"], np.cos(np.deg2rad(aspect)))
            if 'aspect_sin' not in var_in:
                ds['aspect_sin'] = (["y", "x"], np.sin(np.deg2rad(aspect)))

        if 'svf' not in var_in:
            print('Computing svf ...')
            svf = viewf.viewf(np.double(dem_arr), dx)[0]
            ds['svf'] = (["y", "x"], svf)

        ds.attrs = dict(description="DEM input parameters to TopoSub",
                       author="TopoPyScale, https://github.com/ArcticSnow/TopoPyScale")
        ds.x.attrs = {'units': 'm'}
        ds.y.attrs = {'units': 'm'}
        ds.elevation.attrs = {'units': 'm'}
        ds.slope.attrs = {'units': 'rad'}
        ds.aspect.attrs = {'units': 'rad'}
        ds.aspect_cos.attrs = {'units': 'cosinus'}
        ds.aspect_sin.attrs = {'units': 'sinus'}
        ds.svf.attrs = {'units': 'ratio', 'standard_name': 'svf', 'long_name': 'Sky view factor'}

        if file_ds.is_file():
            te.to_netcdf(ds, fname=pdir / output_folder / 'tmp' / fname)
            shutil.move(pdir / output_folder / 'tmp' /fname, file_ds)
            ds = xr.open_dataset(file_ds)
        else:
            te.to_netcdf(ds, fname=file_ds)

    return ds


def compute_horizon(dem_file, azimuth_inc=30, num_threads=None, fname='da_horizon.nc', output_directory=Path('./outputs')):
    """
    Function to compute horizon angles for

    Args:
        dem_file (str): path and filename of the dem
        azimuth_inc (int): angle increment to compute horizons at, in Degrees [0-359]
        num_threads (int): number of threads to parallize on

    Returns: 
        dataarray: all horizon angles for x,y,azimuth coordinates
         
    """
    print('\n---> Computing horizons with {} degree increments'.format(azimuth_inc))
    ds = rioxarray.open_rasterio(dem_file).to_dataset('band')
    ds = ds.rename({1: 'elevation'})
    dx = ds.x.diff('x').median().values

    azimuth = np.arange(-180 + azimuth_inc / 2, 180, azimuth_inc) # center the azimuth in middle of the bin
    arr_val = np.empty((azimuth.shape[0], ds.elevation.shape[0], ds.elevation.shape[1]))

    if num_threads is None:
        pool = ThreadPool(mproc.cpu_count() - 2)
    else:
        pool = ThreadPool(num_threads)

    elev = []
    dxs = []
    for azi in azimuth:
        elev.append(ds.elevation.values)
        dxs.append(dx)

    arr = pool.starmap(horizon.horizon, zip(list(azimuth),
                                                   elev,
                                                   dxs))
    pool.close()
    pool.join()

    for i, a in enumerate(arr):
        arr_val[i,:,:] = a

    da = xr.DataArray(data=np.pi/2 - np.arccos(arr_val),
                      coords={
                          "y": ds.y.values,
                          "x": ds.x.values,
                          "azimuth": azimuth
                      },
                      dims=["azimuth", "y", "x"],
                      name='horizon',
                      attrs={
                          'long_name': 'Horizon angles',
                          'units':'degree'
                      }
                      )
    te.to_netcdf(da.to_dataset(), fname=output_directory / fname)
    return da


