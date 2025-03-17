"""
Retrieve ERA5 data from NCAR RDA (dataset d633000) using THREDDS Data Server.

Based on original implementation for CDS API by:
- J. Fiddes, Origin implementation
- S. Filhol adapted in 2021
"""
import os
import random
import sys
import shutil
import time

import requests
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
from multiprocessing.dummy import Pool as ThreadPool
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from datetime import datetime
from multiprocessing.pool import ThreadPool
from functools import partial


def rda_login(orcid_id, orcid_password,
              login_url="https://api.rda-web-prod01.ucar.edu/login/",
              home_page_url="https://api.rda-web-prod01.ucar.edu/login/"):
    """
    Logs into Research RDA at NCAR using ORCID credentials.

    Args:
        orcid_id (str): Your ORCID ID (typically your email).
        orcid_password (str): Your ORCID password.
        login_url (str, optional): The login URL for Research RDA. Defaults to "https://api.rda-web-prod01.ucar.edu/login/".
        home_page_url (str, optional): The expected home page URL after successful login. Defaults to "https://rda.ucar.edu/".

    Returns:
        requests.Session: A requests Session object if login is successful, allowing you to make authenticated requests.
        None: If login fails.  Prints error messages to the console.
    """

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0'})

    try:
        login_page_response = session.get(login_url)
        login_page_response.raise_for_status()

        soup = BeautifulSoup(login_page_response.text, 'html.parser')
        csrf_input = soup.find('input', {'name': 'csrfmiddlewaretoken'})
        csrf_token = csrf_input.get('value') if csrf_input else None
        if not csrf_input or not csrf_token:
            print("Error: CSRF token not found on login page. Inspect the login page source code.")
            return None

        login_data = {
            'csrfmiddlewaretoken': csrf_token,
            'username': orcid_id,  # Assuming 'username' is the correct field name
            'password': orcid_password,  # Assuming 'password' is the correct field name
        }

        login_submission_response = session.post(
            login_url,
            data=login_data,
            headers={'Referer': login_url},
            allow_redirects=True,
        )
        login_submission_response.raise_for_status()

        if home_page_url in login_submission_response.url:
            print("Login successful!")
            return session  # Return the session object for authenticated requests
        else:
            print("Login failed.")
            print(f"Response status code: {login_submission_response.status_code}")
            print(f"Response URL: {login_submission_response.url}")
            # print("Response content (for debugging):\n", login_submission_response.text) # Uncomment for detailed debugging
            return None  # Indicate login failure

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        print("Failed to retrieve login page or submit login form.")
        return None
    except Exception as e:
        print(f"General Error: {e}")
        return None


class NCAR_RDA_Downloader:
    """Class to handle authentication and downloading from NCAR RDA."""

    def __init__(self, orcid_id=None, orcid_password=None):
        """
        Initialize the downloader with authentication information.

        Args:
            use_netrc: Whether to use .netrc file for credentials
            orcid_id: ORCID ID if not using netrc
            orcid_password: ORCID password if not using netrc
        """
        self.orcid_id = orcid_id
        self.orcid_password = orcid_password
        self.session = requests.Session()
        self.authenticated = False

    def authenticate(self):
        """Authenticate with NCAR RDA using ORCID credentials."""
        if self.authenticated:
            return True

        # --- Usage of the function ---
        self.session = rda_login(self.orcid_id, self.orcid_password)

        if self.session:
            print("Successfully authenticated with NCAR RDA")
            self.authenticated = True
            return True
        else:
            print("Failed to authenticate with NCAR RDA. Please check your credentials.")
            return False


def retrieve_era5_ncar(startDate, endDate, outputDir, latN, latS, lonE, lonW,
                       surf_vars=None, plev_vars=None, plevels=None,
                       time_step='1H', num_threads=10, orcid_id=None, orcid_password=None):
    """
    Sets up ERA5 data retrieval from NCAR RDA.

    Args:
        startDate: Start date for data retrieval (YYYY-MM-DD)
        endDate: End date for data retrieval (YYYY-MM-DD)
        outputDir: Directory to write output files
        latN: North latitude of bounding box
        latS: South latitude of bounding box
        lonE: East longitude of bounding box
        lonW: West longitude of bounding box
        surf_vars: List of surface variables to download
        plev_vars: List of pressure level variables to download
        plevels: List of pressure levels to download
        time_step: Time step to use: '1H', '3H', '6H'
        num_threads: Number of threads to use for downloading
        orcid_id: ORCID ID for authentication
        orcid_password: ORCID password for authentication

    Returns:
        Downloaded ERA5 files stored on disk.
    """
    print('\n---> Loading ERA5 data from NCAR RDA')
    # surface_vars = ['d2m', 't2m', 'sp', 'ssrd', 'strd', 'tp', 'z']
    # Default variables if not specified
    if surf_vars is None:
        surf_vars = {
            'Z': '129',  # Geopotential
            '2D': '168',  # 2 metre dewpoint temperature
            'STRD': '175',  # Surface thermal radiation downwards
            'SSRD': '169',  # Surface solar radiation downwards
            'SP': '134',  # Surface pressure
            'TP': '228',  # Total precipitation
            '2T': '167',  # 2 metre temperature
            #'TISR': '212',  # TOA incident solar radiation
            #'ZUST': '228003',  # Friction velocity
            #'IE': '232',  # instantaneous_moisture_flux
            #'ISHF': '231'  # instantaneous_surface_sensible_heat_flux
        }

    if plev_vars is None:
        plev_vars = {
            'Z': '129',  # Geopotential
            'T': '130',  # Temperature
            'U': '131',  # U component of wind
            'V': '132',  # V component of wind
            'R': '157',  # Relative humidity
            'Q': '133'  # Specific humidity
        }

    if plevels is None:
        plevels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']

    # Time steps dictionary
    time_step_dict = {
        '1H': list(f"{h:02d}:00" for h in range(24)),
        '3H': ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
        '6H': ['00:00', '06:00', '12:00', '18:00']
    }

    # Make sure output directories exist
    os.makedirs(outputDir, exist_ok=True)
    surf_dir = os.path.join(outputDir, 'surf')
    plev_dir = os.path.join(outputDir, 'plev')
    os.makedirs(surf_dir, exist_ok=True)
    os.makedirs(plev_dir, exist_ok=True)

    # Create date range for downloads
    df = pd.DataFrame()
    df['dates'] = pd.date_range(startDate, endDate, freq='M')
    df['month'] = df.dates.dt.month
    df['year'] = df.dates.dt.year

    # Add metadata for surface files
    df['surf_target_file'] = df.dates.apply(
        lambda x: os.path.join(surf_dir, f"SURF_{x.year:04d}{x.month:02d}.nc")
    )
    df['surf_file_exist'] = df.surf_target_file.apply(lambda x: os.path.isfile(x) * 1)

    # Add metadata for pressure level files
    df['plev_target_file'] = df.dates.apply(
        lambda x: os.path.join(plev_dir, f"PLEV_{x.year:04d}{x.month:02d}.nc")
    )
    df['plev_file_exist'] = df.plev_target_file.apply(lambda x: os.path.isfile(x) * 1)

    # Add time steps
    df['time_steps'] = df.apply(lambda x: time_step_dict.get(time_step), axis=1)

    # Add bounding box
    bbox = [latN, lonW, latS, lonE]
    df['bbox'] = df.apply(lambda x: bbox, axis=1)

    # Initialize NCAR RDA downloader
    downloader = NCAR_RDA_Downloader(orcid_id=orcid_id, orcid_password=orcid_password)
    auth_success = downloader.authenticate()
    if not auth_success:
        sys.exit("Authentication failed. Please check your credentials.")

    print(f"Start date = {df.dates[0].strftime('%Y-%m-%d')}")
    print(f"End date = {df.dates[len(df.dates) - 1].strftime('%Y-%m-%d')}")

    # Report on existing files
    if df.surf_file_exist.sum() > 0:
        print("Surface data files already found:")
        print(df.surf_target_file.loc[df.surf_file_exist == 1].apply(lambda x: os.path.basename(x)).tolist())

    if df.plev_file_exist.sum() > 0:
        print("Pressure level data files already found:")
        print(df.plev_target_file.loc[df.plev_file_exist == 1].apply(lambda x: os.path.basename(x)).tolist())

    # Process surface data downloads
    # surf_download = df.loc[df.surf_file_exist == 0]
    # if surf_download.shape[0] > 0:
    #     print("Downloading surface data from NCAR RDA:")
    #     print(surf_download.surf_target_file.apply(lambda x: os.path.basename(x)).tolist())
    #
    #     pool = ThreadPool(num_threads)
    #     pool.starmap(
    #         download_surface_data,
    #         zip(
    #             [downloader.session] * len(surf_download),
    #             list(surf_download.year),
    #             list(surf_download.month),
    #             list(surf_download.surf_target_file),
    #             list(surf_download.bbox),
    #             [surf_vars] * len(surf_download),
    #             list(surf_download.time_steps)
    #         )
    #     )
    #     pool.close()
    #     pool.join()

    # Process pressure level data downloads
    plev_download = df.loc[df.plev_file_exist == 0]
    if plev_download.shape[0] > 0:
        print("Downloading pressure level data from NCAR RDA:")
        print(plev_download.plev_target_file.apply(lambda x: os.path.basename(x)).tolist())

        pool = ThreadPool(num_threads)
        pool.starmap(
            download_pressure_data,
            zip(
                [downloader.session] * len(plev_download),
                list(plev_download.year),
                list(plev_download.month),
                list(plev_download.plev_target_file),
                list(plev_download.bbox),
                [plev_vars] * len(plev_download),
                [plevels] * len(plev_download),
            )
        )
        pool.close()
        pool.join()

    return {
        'surface_files': df.surf_target_file.tolist(),
        'plevel_files': df.plev_target_file.tolist()
    }


def download_surface_data(session, year, month, target_file, bbox, variables, time_steps):
    """
    Download ERA5 surface data from NCAR RDA THREDDS server.

    Args:
        session: Requests session with authentication
        year: Year to download
        month: Month to download
        target_file: Output file path
        bbox: Bounding box [latN, lonW, latS, lonE]
        variables: Dictionary of variable names and their GRIB codes
        time_steps: List of time steps to download in HH:MM format

    Returns:
        Downloaded and merged data saved to target_file, True if successful, False otherwise.
    """
    print(f"Downloading ERA5 surface data for {year}-{month:02d}...")

    # Create temporary directory for variable downloads
    temp_dir = f"{target_file}_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Calculate days in month for file naming
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    days_in_month = (next_month - datetime(year, month, 1)).days

    month_str = f"{year}{month:02d}"
    start_date = f"{year}-{month:02d}-01T00:00:00Z"
    end_date = f"{year}-{month:02d}-{days_in_month:02d}T23:00:00Z"

    # Special handling for geopotential (Z) which needs to be fetched from pressure level data
    z_variable = None
    if 'Z' in variables:
        z_grib_code = variables.pop('Z')  # Remove Z from surface variables
        print(f"  Note: Geopotential (Z) will be obtained from pressure level data at 1000 hPa")
        z_variable = {'Z': z_grib_code}

    # Special handling for accumulated variables (STRD, SSRD)
    accumulated_vars = {}
    tp_calculation_needed = False

    for var in ['STRD', 'SSRD']:
        if var in variables:
            accumulated_vars[var] = variables.pop(var)
            print(f"  Note: {var} will be downloaded from accumulated forecast data")

    # Handle TP (Total Precipitation) which needs to be calculated from LSP and CP
    if 'TP' in variables:
        print("  Note: TP (Total Precipitation) will be calculated from LSP and CP")
        variables.pop('TP')  # Remove TP from regular variables
        tp_calculation_needed = True
        # Make sure LSP and CP are not in the regular variables
        if 'LSP' in variables:
            variables.pop('LSP')
        if 'CP' in variables:
            variables.pop('CP')
        # Add LSP and CP to accumulated vars with their codes
        accumulated_vars['LSP'] = '142'
        accumulated_vars['CP'] = '143'

    # Download regular surface variables
    variable_files = []
    for var_abbrev, grib_code in variables.items():
        var_file = os.path.join(temp_dir, f"{var_abbrev}.nc")
        print(f"  Downloading variable: {var_abbrev} (GRIB code: {grib_code})")

        # Construct proper filename based on provided examples
        filename = f"e5.oper.an.sfc.128_{grib_code}_{var_abbrev.lower()}.ll025sc.{month_str}0100_{month_str}{days_in_month:02d}23.nc"

        # Build the NCSS URL (using grid path as shown in your examples)
        base_url = f"https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d633000/e5.oper.an.sfc/{month_str}/{filename}"

        # Parameters for the request
        # For variables starting with a number, prefix with "VAR_"
        var_param = f"VAR_{var_abbrev}" if var_abbrev[0].isdigit() else var_abbrev

        params = {
            'var': var_param,
            'north': bbox[0],
            'west': bbox[1],
            'south': bbox[2],
            'east': bbox[3],
            'horizStride': 1,
            'time_start': start_date,
            'time_end': end_date,
            'accept': 'netcdf4-classic',
            'addLatLon': 'true'
        }

        # Add time filtering if specified
        # if time_steps and time_steps != ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
        #                                  '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
        #                                  '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']:
        #     # If not all hours, we need to specify the times
        #     time_param = []
        #     for ts in time_steps:
        #         for day in range(1, days_in_month + 1):
        #             time_param.append(f"{year}-{month:02d}-{day:02d}T{ts}Z")
        #     params['time'] = time_param

        # Build the full URL
        url = f"{base_url}?{urlencode(params, doseq=True)}"
        print(f"  Request URL: {url}")

        try:
            response = session.get(url, stream=True)
            if response.status_code == 200:
                with open(var_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                variable_files.append(var_file)
                print(f"    Downloaded {var_abbrev} for {year}-{month:02d}")
            else:
                print(f"    Failed to download {var_abbrev} for {year}-{month:02d}: {response.status_code}")
                print(f"    Response: {response.text[:500]}...")
        except requests.exceptions.RequestException as e:
            print(f"    Request Error for {var_abbrev} in {year}-{month:02d}: {e}")
            continue

    # Download accumulated variables (split in two files per month)
    for var_abbrev, grib_code in accumulated_vars.items():
        var_file = os.path.join(temp_dir, f"{var_abbrev}.nc")
        print(f"  Downloading accumulated variable: {var_abbrev} (GRIB code: {grib_code})")

        # Calculate the split dates (day 1 to day 15, then day 16 to next month day 1)
        # First half: day 1 hour 6 to day 15 hour 18
        first_half_start = f"{year}-{month:02d}-01T06:00:00Z"
        first_half_end = f"{year}-{month:02d}-15T18:00:00Z"
        first_half_filename = f"e5.oper.fc.sfc.accumu.128_{grib_code}_{var_abbrev.lower()}.ll025sc.{month_str}0106_{month_str}1606.nc"

        # Second half: day 16 hour 6 to next month day 1 hour 6
        second_half_start = f"{year}-{month:02d}-16T06:00:00Z"

        # Calculate next month string
        if month == 12:
            next_month_str = f"{year + 1}0101"
        else:
            next_month_str = f"{year}{month + 1:02d}0106"

        second_half_end = f"{next_month_str[0:4]}-{next_month_str[4:6]}-{next_month_str[6:8]}T{next_month_str[8:10]}:00:00Z"
        second_half_filename = f"e5.oper.fc.sfc.accumu.128_{grib_code}_{var_abbrev.lower()}.ll025sc.{month_str}1606_{next_month_str}.nc"

        # Temporary files for both halves
        first_half_file = os.path.join(temp_dir, f"{var_abbrev}_first_half.nc")
        second_half_file = os.path.join(temp_dir, f"{var_abbrev}_second_half.nc")

        # Download first half
        base_url = f"https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d633000/e5.oper.fc.sfc.accumu/{month_str}/{first_half_filename}"

        # For variables starting with a number, prefix with "VAR_"
        var_param = f"VAR_{var_abbrev}" if var_abbrev[0].isdigit() else var_abbrev

        params = {
            'var': var_param,
            'north': bbox[0],
            'west': bbox[1],
            'south': bbox[2],
            'east': bbox[3],
            'horizStride': 1,
            'time_start': first_half_start,
            'time_end': first_half_end,
            'accept': 'netcdf4-classic',
            'addLatLon': 'true'
        }

        # Add time filtering if specified (adapted for 6-hourly data)
        # if time_steps and any(ts in time_steps for ts in ['06:00', '12:00', '18:00', '00:00']):
        #     time_param = []
        #     for ts in time_steps:
        #         if ts in ['06:00', '12:00', '18:00', '00:00']:  # Only use 6-hourly steps
        #             for day in range(1, 17):  # Days 1-16
        #                 time_param.append(f"{year}-{month:02d}-{day:02d}T{ts}Z")
        #     if time_param:
        #         params['time'] = time_param

        # Build the full URL
        url = f"{base_url}?{urlencode(params, doseq=True)}"
        print(f"  Request URL for {var_abbrev} (first half): {url}")

        try:
            response = session.get(url, stream=True)
            if response.status_code == 200:
                with open(first_half_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"    Downloaded {var_abbrev} first half for {year}-{month:02d}")

                # Download second half
                base_url = f"https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d633000/e5.oper.fc.sfc.accumu/{month_str}/{second_half_filename}"

                params['time_start'] = second_half_start
                params['time_end'] = second_half_end

                # Update time filtering for second half if needed
                # if time_steps and any(ts in time_steps for ts in ['06:00', '12:00', '18:00', '00:00']):
                #     time_param = []
                #     for ts in time_steps:
                #         if ts in ['06:00', '12:00', '18:00', '00:00']:  # Only use 6-hourly steps
                #             for day in range(16, days_in_month + 1):  # Days 16 to end of month
                #                 time_param.append(f"{year}-{month:02d}-{day:02d}T{ts}Z")
                #     if time_param:
                #         params['time'] = time_param

                url = f"{base_url}?{urlencode(params, doseq=True)}"
                print(f"  Request URL for {var_abbrev} (second half): {url}")

                response = session.get(url, stream=True)
                if response.status_code == 200:
                    with open(second_half_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(f"    Downloaded {var_abbrev} second half for {year}-{month:02d}")

                    # Combine both halves
                    try:
                        # Open both datasets
                        first_half_ds = xr.open_dataset(first_half_file)
                        second_half_ds = xr.open_dataset(second_half_file)

                        # Combine along time dimension
                        combined_ds = xr.concat([first_half_ds, second_half_ds], dim='time')

                        # Save combined dataset
                        combined_ds.to_netcdf(var_file)

                        # Close datasets
                        first_half_ds.close()
                        second_half_ds.close()

                        # Add to list of variable files
                        variable_files.append(var_file)
                        print(f"    Combined {var_abbrev} data for {year}-{month:02d}")

                        # Clean up temporary files
                        os.remove(first_half_file)
                        os.remove(second_half_file)
                    except Exception as e:
                        print(f"    Error combining {var_abbrev} data: {str(e)}")
                else:
                    print(
                        f"    Failed to download {var_abbrev} second half for {year}-{month:02d}: {response.status_code}")
                    print(f"    Response: {response.text[:500]}...")
            else:
                print(f"    Failed to download {var_abbrev} first half for {year}-{month:02d}: {response.status_code}")
                print(f"    Response: {response.text[:500]}...")
        except requests.exceptions.RequestException as e:
            print(f"    Request Error for {var_abbrev} in {year}-{month:02d}: {e}")
            continue

    # Download geopotential (Z) from pressure level data if requested
    if z_variable:
        z_var_abbrev, z_grib_code = list(z_variable.items())[0]
        z_file = os.path.join(temp_dir, f"{z_var_abbrev}_surface.nc")
        print(f"  Downloading geopotential (Z) from pressure level data at 1000 hPa")

        # Construct filename for pressure level data
        filename = f"e5.oper.an.pl.128_{z_grib_code}_{z_var_abbrev.lower()}.ll025sc.{month_str}0100_{month_str}{days_in_month:02d}23.nc"

        # Build the NCSS URL for pressure level data
        base_url = f"https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d633000/e5.oper.an.pl/{month_str}/{filename}"

        # Parameters for the request - specify 1000 hPa level
        # For variables starting with a number, prefix with "VAR_"
        var_param = f"VAR_{z_var_abbrev}" if z_var_abbrev[0].isdigit() else z_var_abbrev

        params = {
            'var': var_param,
            'north': bbox[0],
            'west': bbox[1],
            'south': bbox[2],
            'east': bbox[3],
            'horizStride': 1,
            'time_start': start_date,
            'time_end': end_date,
            'vertCoord': '1000',  # 1000 hPa level as sea level approximation
            'accept': 'netcdf4-classic',
            'addLatLon': 'true'
        }

        # Add time filtering if specified
        # if time_steps and time_steps != ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
        #                                  '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
        #                                  '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']:
        #     time_param = []
        #     for ts in time_steps:
        #         for day in range(1, days_in_month + 1):
        #             time_param.append(f"{year}-{month:02d}-{day:02d}T{ts}Z")
        #     params['time'] = time_param

        # Build the full URL
        url = f"{base_url}?{urlencode(params, doseq=True)}"
        print(f"  Request URL for Z: {url}")

        try:
            response = session.get(url, stream=True)
            if response.status_code == 200:
                with open(z_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Process the Z file to remove level dimension and rename it
                try:
                    # Open the dataset
                    z_ds = xr.open_dataset(z_file)

                    # If there's a level dimension, select just the 1000 hPa level
                    if 'level' in z_ds.dims:
                        z_ds = z_ds.sel(level=1000, drop=True)

                    # Save the modified dataset
                    z_ds.to_netcdf(z_file + ".processed")
                    z_ds.close()

                    # Replace the original file with the processed one
                    os.remove(z_file)
                    os.rename(z_file + ".processed", z_file)

                    variable_files.append(z_file)
                    print(f"    Downloaded and processed geopotential (Z) at 1000 hPa for {year}-{month:02d}")
                except Exception as e:
                    print(f"    Error processing geopotential (Z) file: {str(e)}")
            else:
                print(f"    Failed to download geopotential (Z) for {year}-{month:02d}: {response.status_code}")
                print(f"    Response: {response.text[:500]}...")
        except requests.exceptions.RequestException as e:
            print(f"    Request Error for geopotential (Z) in {year}-{month:02d}: {e}")

    # Merge the downloaded variables into a single file
    if variable_files:
        try:
            datasets = [xr.open_dataset(file) for file in variable_files]

            # If we need to calculate TP (Total Precipitation)
            if tp_calculation_needed:
                print("  Calculating TP (Total Precipitation) from LSP and CP")
                # Find the LSP and CP datasets
                lsp_ds = None
                cp_ds = None

                for ds in datasets[:]:
                    if 'LSP' in ds.data_vars:
                        lsp_ds = ds
                    elif 'CP' in ds.data_vars:
                        cp_ds = ds

                if lsp_ds is not None and cp_ds is not None:
                    # Create a new dataset with TP variable
                    tp_ds = xr.Dataset()

                    # Calculate TP = LSP + CP
                    tp_ds['TP'] = lsp_ds['LSP'] + cp_ds['CP']

                    # Copy coordinates and attributes
                    for coord in lsp_ds.coords:
                        tp_ds[coord] = lsp_ds[coord]

                    # Set attributes for the TP variable
                    tp_ds.TP.attrs = {
                        'long_name': 'Total Precipitation',
                        'units': lsp_ds.LSP.attrs.get('units', 'm'),
                        'standard_name': 'total_precipitation',
                        'calculated': 'Sum of Large Scale Precipitation (LSP) and Convective Precipitation (CP)'
                    }

                    # Add TP dataset to the list
                    datasets.append(tp_ds)
                    print("  Successfully created TP variable")

                    # Remove LSP and CP variables if TP was successfully created
                    # Keep the actual datasets for their coordinate information
                    for i, ds in enumerate(datasets):
                        if 'LSP' in ds.data_vars:
                            datasets[i] = ds.drop_vars('LSP')
                        elif 'CP' in ds.data_vars:
                            datasets[i] = ds.drop_vars('CP')
                else:
                    print("  Warning: Could not calculate TP - LSP or CP missing")

            merged_ds = xr.merge(datasets)
            merged_ds.to_netcdf(target_file)
            for ds in datasets:
                ds.close()
            print(f"Merged variables into {target_file}")
            shutil.rmtree(temp_dir)
            return True
        except Exception as e:
            print(f"Error merging variables: {str(e)}")
            return False
    else:
        print(f"No variables downloaded for {year}-{month:02d}")
        return False


def download_pressure_data(session, year, month, target_file, bbox, variables, plevels, max_workers=6,
                           initial_retry_delay=30):
    """
    Download ERA5 pressure level data from NCAR RDA THREDDS server using parallel processing.

    Args:
        session: Requests session with authentication
        year: Year to download
        month: Month to download
        target_file: Output file path
        bbox: Bounding box [latN, lonW, latS, lonE]
        variables: Dictionary of pressure level variables to download
        plevels: List of pressure levels to download (filtered during download)
        max_workers: Maximum number of concurrent download workers
        initial_retry_delay: Initial delay in seconds before retrying failed downloads

    Returns:
        Downloaded and merged data saved to target_file, True if successful, False otherwise.
    """
    print(f"Downloading ERA5 pressure level data for {year}-{month:02d} using parallel processing...")
    month_start_time = time.time()
    # Create temporary directory for variable downloads
    temp_dir = f"{target_file}_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Calculate days in month
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    days_in_month = (next_month - datetime(year, month, 1)).days

    month_str = f"{year}{month:02d}"

    # Function to download a single time part for a day-variable combination
    def download_day_variable_part(day, var_abbrev, grib_code, time_part, initial_delay=30):
        day_str = f"{day:02d}"

        # Define time ranges for the three 8-hour periods
        time_ranges = [
            (0, 7),  # 00:00 - 07:00
            (8, 15),  # 08:00 - 15:00
            (16, 23)  # 16:00 - 23:00
        ]

        start_hour, end_hour = time_ranges[time_part]
        start_date = f"{year}-{month:02d}-{day:02d}T{start_hour:02d}:00:00Z"
        end_date = f"{year}-{month:02d}-{day:02d}T{end_hour:02d}:00:00Z"

        # Define file path for this time part
        part_file = os.path.join(temp_dir, f"{var_abbrev}_{year}{month:02d}{day:02d}_part{time_part}.nc")

        # Construct proper filename for daily file
        filename = f"e5.oper.an.pl.128_{grib_code}_{var_abbrev.lower()}.ll025{'sc' if var_abbrev.lower() not in ('u', 'v') else 'uv'}.{year}{month:02d}{day:02d}00_{year}{month:02d}{day:02d}23.nc"

        # Build the NCSS URL
        base_url = f"https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d633000/e5.oper.an.pl/{month_str}/{filename}"

        # Parameters for the request - include pressure level filter
        params = {
            'var': var_abbrev,
            'north': bbox[0],
            'west': bbox[1],
            'south': bbox[2],
            'east': bbox[3],
            'horizStride': 1,
            'time_start': start_date,
            'time_end': end_date,
            'accept': 'netcdf4-classic',
            'addLatLon': 'true',
            # 'vertCoord': plevels  # Only download requested pressure levels
        }

        # Build the full URL
        url = f"{base_url}?{urlencode(params, doseq=True)}"

        # Implement retry logic with exponential backoff
        retry_count = 0
        delay = initial_delay

        while True:  # Keep trying until successful
            try:
                # If this is a retry, log it
                if retry_count > 0:
                    print(
                        f"  Retry #{retry_count} for {var_abbrev} (hours {start_hour}-{end_hour}) for {year}-{month:02d}-{day:02d}")
                else:
                    print(
                        f"  Downloading {var_abbrev} (hours {start_hour}-{end_hour}) for {year}-{month:02d}-{day:02d}")

                response = session.get(url, stream=True)

                if response.status_code == 200:
                    with open(part_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(
                        f"    Successfully downloaded {var_abbrev} (hours {start_hour}-{end_hour}) for {year}-{month:02d}-{day:02d}")
                    return (var_abbrev, day, time_part, part_file)
                else:
                    # Any non-200 response will be retried
                    print(
                        f"    Failed to download {var_abbrev} (hours {start_hour}-{end_hour}) for {year}-{month:02d}-{day:02d}: {response.status_code}")

                    # Log more details if it's a server error
                    if response.status_code == 500:
                        if "OutOfMemoryError" in response.text:
                            print(f"    Server out of memory error detected")
                        else:
                            print(f"    Server error: {response.text[:200]}...")

                    print(f"    Waiting {delay} seconds before retrying (retry #{retry_count + 1})")
                    time.sleep(delay)
                    # Exponential backoff with jitter
                    delay = min(delay * 1.5 + random.uniform(1, 5), 300)  # Cap at 5 minutes
                    retry_count += 1

            except requests.exceptions.RequestException as e:
                print(
                    f"    Request Error for {var_abbrev} (hours {start_hour}-{end_hour}) in {year}-{month:02d}-{day:02d}: {e}")
                print(f"    Waiting {delay} seconds before retrying (retry #{retry_count + 1})")
                time.sleep(delay)
                # Exponential backoff with jitter
                delay = min(delay * 1.5 + random.uniform(1, 5), 300)  # Cap at 5 minutes
                retry_count += 1

    # Function to combine time parts for a single day-variable
    def combine_day_parts(var_abbrev, day, part_files, max_retries=3):
        if not part_files:
            print(f"  No parts available for {var_abbrev} on day {day}, cannot combine")
            return None

        if None in part_files:
            # Filter out None values and warn about missing parts
            valid_parts = [p for p in part_files if p is not None]
            missing_parts = [i for i, p in enumerate(part_files) if p is None]
            print(f"  Missing parts {missing_parts} for {var_abbrev} on day {day}, combining available parts only")
            part_files = valid_parts

        if not part_files:
            print(f"  No valid parts for {var_abbrev} on day {day} after filtering")
            return None

        try:
            # Output file for the full day
            daily_file = os.path.join(temp_dir, f"{var_abbrev}_{year}{month:02d}{day:02d}.nc")

            # Open and combine the part files
            datasets = [xr.open_dataset(file) for file in part_files]
            combined_ds = xr.concat(datasets, dim='time')
            combined_ds.to_netcdf(daily_file)

            # Close and remove part files
            for ds, file in zip(datasets, part_files):
                ds.close()
                os.remove(file)

            print(f"  Combined all time parts for {var_abbrev} on {year}-{month:02d}-{day:02d}")
            return daily_file
        except Exception as e:
            print(f"  Error combining time parts for {var_abbrev} on day {day}: {str(e)}")
            return None

    # Create a list of all day-variable-part combinations to download
    download_tasks = []
    for day in range(1, days_in_month + 1):
        for var_abbrev, grib_code in variables.items():
            for time_part in range(3):  # 3 parts per day
                # print(f"Tasks: {var_abbrev}, {day}, {time_part}")
                download_tasks.append((day, var_abbrev, grib_code, time_part, initial_retry_delay))

    # Dictionary to store part files for each day-variable combination
    day_var_parts = {}

    # Dictionary to track retry attempts for each part
    part_attempts = {}

    # Dictionary to store file paths for each variable for each day
    var_daily_files = {var_abbrev: [] for var_abbrev in variables}

    # Set to track completed downloads to avoid duplicate processing
    completed_downloads = set()

    # Use ThreadPool instead of ThreadPoolExecutor
    print(f"Starting parallel downloads with {max_workers} workers...")
    with ThreadPool(processes=max_workers) as pool:
        results = pool.starmap(download_day_variable_part, download_tasks)

        # Process results
        for result in results:
            if result is not None:
                var, day_num, part_num, file_path = result

                # Track this attempt for the part
                key = (var, day_num)
                part_key = (key, part_num)
                if part_key not in part_attempts:
                    part_attempts[part_key] = 0
                part_attempts[part_key] += 1

                if file_path:
                    # Check if this key has already been processed
                    if key in completed_downloads:
                        print(f"  Already processed {var} for day {day_num}, skipping...")
                        continue

                    if key not in day_var_parts:
                        day_var_parts[key] = [None, None, None]  # Initialize for 3 parts
                    day_var_parts[key][part_num] = file_path

                    # Check if we have all attempted parts for this day-variable
                    all_attempted = all(part_attempt == 3 for part_attempt in
                                        [(1 if p is not None else 0) + part_attempts.get((key, i), 0) for i, p in
                                         enumerate(day_var_parts[key])])

                    # Proceed with combining if we have all parts or have tried each part enough times
                    if all(part is not None for part in day_var_parts[key]) or all_attempted:
                        print(f"  Processing {var} on day {day_num}, combining available parts...")
                        daily_file = combine_day_parts(var, day_num, day_var_parts[key])
                        if daily_file:
                            var_daily_files[var].append(daily_file)
                            # Mark this key as completed
                            completed_downloads.add(key)
                        # Remove this key from the tracking dictionary
                        if key in day_var_parts:
                            del day_var_parts[key]

    # Process any remaining day-variable combinations
    if day_var_parts:
        print(f"Attempting to process {len(day_var_parts)} remaining day-variable combinations with partial data...")
        for (var, day), parts in day_var_parts.items():
            if (var, day) not in completed_downloads:
                print(f"  Processing remaining {var} on day {day} with available parts...")
                daily_file = combine_day_parts(var, day, parts)
                if daily_file:
                    var_daily_files[var].append(daily_file)

    # Now combine daily files for each variable
    print("Combining daily files for each variable...")
    var_combined_files = {}

    for var_abbrev, daily_files in var_daily_files.items():
        if not daily_files:
            print(f"  No daily files to combine for {var_abbrev}")
            continue

        combined_file = os.path.join(temp_dir, f"{var_abbrev}_combined.nc")

        try:
            # Open all daily files for this variable
            print(f"  Combining {len(daily_files)} daily files for {var_abbrev}")
            datasets = [xr.open_dataset(file) for file in daily_files]

            # Combine along time dimension
            combined_ds = xr.concat(datasets, dim='time')
            combined_ds.to_netcdf(combined_file)

            # Close datasets
            for ds in datasets:
                ds.close()

            var_combined_files[var_abbrev] = combined_file
            print(f"  Combined daily files for {var_abbrev}")

            # Clean up daily files
            for file in daily_files:
                os.remove(file)

        except Exception as e:
            print(f"  Error combining daily files for {var_abbrev}: {str(e)}")
            # If combining fails, we'll try to use the first daily file
            var_combined_files[var_abbrev] = daily_files[0] if daily_files else None

        month_end_time = time.time()

    elapsed_time = month_end_time - month_start_time

    print(f"Downloaded all PL {month_str} data in {elapsed_time:.2f} seconds.")

    # Merge all variables into a single file
    print("Merging all variables into final file...")
    var_files = [file for file in var_combined_files.values() if file is not None]
    if var_files:
        try:
            datasets = [xr.open_dataset(file) for file in var_files]
            merged_ds = xr.merge(datasets)
            merged_ds.to_netcdf(target_file)
            for ds in datasets:
                ds.close()
            print(f"Merged all variables into {target_file}")
            shutil.rmtree(temp_dir)
            return True
        except Exception as e:
            print(f"Error merging variables: {str(e)}")
            return False
    else:
        print(f"No variables downloaded for {year}-{month:02d}")
        shutil.rmtree(temp_dir)
        return False


def download_realtime_data(outputDir, latN, latS, lonE, lonW,
                           surf_vars=None, plev_vars=None, plevels=None,
                           time_step='1H', orcid_id=None, orcid_password=None):
    """
    Download the most recent available data for realtime applications.

    Args:
        outputDir: Directory to write output files
        latN, latS, lonE, lonW: Bounding box coordinates
        surf_vars: List of surface variables to download
        plev_vars: List of pressure level variables to download
        plevels: List of pressure levels to download
        time_step: Time step to use
        orcid_id: ORCID ID for authentication
        orcid_password: ORCID password for authentication

    Returns:
        Downloaded realtime data files
    """
    # Find the most recent available date (typically 5 days before current date)
    current_date = datetime.utcnow()
    latest_available = current_date - timedelta(days=5)

    # Download the most recent month
    start_date = datetime(latest_available.year, latest_available.month, 1)
    end_date = latest_available

    print(f"Downloading realtime data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Create realtime directory
    realtime_dir = os.path.join(outputDir, 'realtime')
    os.makedirs(realtime_dir, exist_ok=True)

    # Download data
    return retrieve_era5_ncar(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        realtime_dir,
        latN, latS, lonE, lonW,
        surf_vars, plev_vars, plevels,
        time_step,
        num_threads=1,  # Single thread for realtime
        orcid_id=orcid_id,
        orcid_password=orcid_password
    )


if __name__ == "__main__":
    # Example usage
    start_date = "1995-01-01"
    end_date = "2025-01-31"
    output_dir = "/mnt/datos/rnebot/genesis/DownscalingTopoPyScale/ERA5_Downscaling_Canarias/RDA/"

    # Geographic region (Europe)
    lat_north = 29.5
    lat_south = 27.5
    lon_east = -13.25
    lon_west = -18.5

    # Surface variables to download
    # surf_vars = ['2t', '2d', 'sp', 'tp']
    surf_vars = None

    # Pressure level variables and levels
    plev_vars = None  # ['t', 'z', 'q']
    plevels = None  # ['850', '500', '250']

    # Download the data
    files = retrieve_era5_ncar(
        start_date, end_date, output_dir,
        lat_north, lat_south, lon_east, lon_west,
        surf_vars, plev_vars, plevels,
        time_step='6H',
        num_threads=1,
        orcid_id="rnebot@gmail.com", orcid_password=""
    )

    print("Downloaded files:")
    for key, file_list in files.items():
        print(f"{key}:")
        for file in file_list:
            print(f"  {file}")