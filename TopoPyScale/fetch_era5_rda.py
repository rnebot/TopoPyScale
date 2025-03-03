"""
Retrieve ERA5 data from NCAR RDA (dataset d633000) using THREDDS Data Server.

Based on original implementation for CDS API by:
- J. Fiddes, Origin implementation
- S. Filhol adapted in 2021
"""
import os
import sys
import glob
import shutil
import zipfile
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import xarray as xr
from multiprocessing.dummy import Pool as ThreadPool
from urllib.parse import urlencode
from netrc import netrc
from getpass import getpass
from bs4 import BeautifulSoup


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
            'TISR': '212',  # TOA incident solar radiation
            'ZUST': '228003',  # Friction velocity
            'IE': '232',  # instantaneous_moisture_flux
            'ISHF': '231'  # instantaneous_surface_sensible_heat_flux
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
    surf_download = df.loc[df.surf_file_exist == 0]
    if surf_download.shape[0] > 0:
        print("Downloading surface data from NCAR RDA:")
        print(surf_download.surf_target_file.apply(lambda x: os.path.basename(x)).tolist())

        pool = ThreadPool(num_threads)
        pool.starmap(
            download_surface_data,
            zip(
                [downloader.session] * len(surf_download),
                list(surf_download.year),
                list(surf_download.month),
                list(surf_download.surf_target_file),
                list(surf_download.bbox),
                [surf_vars] * len(surf_download),
                list(surf_download.time_steps)
            )
        )
        pool.close()
        pool.join()

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
                list(plev_download.time_steps)
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
    Download ERA5 data from NCAR RDA OPeNDAP server, allowing variable and spatial/temporal subsetting.

    Args:
        session: Requests session with authentication
        year: Year to download
        month: Month to download
        target_file: Output file path
        bbox: Bounding box [latN, lonW, latS, lonE]
        variables: Dictionary of variable names and their GRIB codes to download (e.g., {'ALNIP': '17', '2T': '167'}).
                   Keys are variable abbreviations (e.g., 'ALNIP', '2T'), values are GRIB codes (e.g., '17', '167').
        time_steps: List of time steps to download in HH:MM format (e.g., ['00:00', '12:00']).
                    Set to None or empty list to download all time steps for the month.

    Returns:
        Downloaded and merged data saved to target_file, True if successful, False otherwise.
    """
    print(f"Downloading ERA5 data for {year}-{month:02d}...")

    # Create temporary directory for variable downloads
    temp_dir = f"{target_file}_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Calculate number of days in the month
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    days_in_month = (next_month - datetime(year, month, 1)).days

    variable_files = []
    for var_abbrev, grib_code in variables.items(): # Iterate through variables dictionary
        var_file = os.path.join(temp_dir, f"{var_abbrev}.nc")
        print(f"  Downloading variable: {var_abbrev} (GRIB code: {grib_code})")

        # Construct DODS URL - using file-based path from HTML source
        # Example path from HTML: files/g/d633000/e5.oper.an.sfc/201207/e5.oper.an.sfc.128_017_alnip.ll025sc.2012070100_2012073123.nc
        # Generalizing the path for year, month, and variable
        month_str = f"{year}{month:02d}"
        filename_base = f"e5.oper.an.sfc.128_{grib_code}_{var_abbrev.lower()}.ll025sc" # Using GRIB code and variable abbreviation
        filename = f"{filename_base}.{month_str}0100_{month_str}{days_in_month:02d}23.nc" # Constructing filename for the whole month
        dods_dataset_path = f"files/g/d633000/e5.oper.an.sfc/{month_str}/{filename}" # Dataset path in THREDDS

        base_url = f"https://thredds.rda.ucar.edu/thredds/dodsC/g/{dods_dataset_path}" # Base DODS URL

        # Construct the projection string (variables and spatial/temporal subsetting)
        projection = ""
        if variables:
            projection_vars = []
            for v_abbrev in variables: # Iterate through variable abbreviations for projection
                var_projection = v_abbrev  # Start with variable abbreviation
                # Add spatial subsetting if bbox is provided
                if bbox:
                    lat_range = f"latitude[{bbox[0]}:{bbox[2]}]"  # North to South (start:stop) - assuming latitude is first dimension
                    lon_range = f"longitude[{bbox[1]}:{bbox[3]}]" # West to East (start:stop) - assuming longitude is second dimension
                    var_projection += f"[{lat_range}][{lon_range}]" # Append spatial subsetting to variable projection
                projection_vars.append(var_projection)
            projection = ",".join(projection_vars) # Comma-separated variables in projection

        # Construct the selection string (temporal subsetting)
        selection = ""
        if time_steps:
            time_indices = []
            # Find time indices corresponding to requested time_steps (assuming hourly data for now)
            hours = [int(ts.split(':')[0]) for ts in time_steps]
            time_indices_str = ','.join([str(h) for h in hours]) # Example: time[0],time[6],time[12]... - assuming time is the first dimension
            time_selection = f"time[{time_indices_str}]" # Time dimension selection
            projection_with_time = []
            for v_proj in projection_vars:
                 projection_with_time.append(f"{v_proj}[{time_selection}][:]") # Add time dimension to each variable projection, spatial dimensions already included
            projection = ",".join(projection_with_time)


        # Construct the full DODS URL with projection
        if projection:
            full_url = f"{base_url}?{projection}"  # .dods
        else:
            full_url = f"{base_url}" # No projection, download all variables... .dods

        print(f"Request URL: {full_url}") # Print full URL for debugging

        try:
            response = session.get(full_url, stream=True)
            if response.status_code == 200:
                with open(var_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                variable_files.append(var_file)
                print(f"    Downloaded {var_abbrev} for {year}-{month:02d}")
            else:
                print(f"    Failed to download {var_abbrev} for {year}-{month:02d}: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"    Request Error for {var_abbrev} in {year}-{month:02d}: {e}")
            continue # Continue to next variable even if one fails


    # Merge the downloaded variables into a single file
    if variable_files:
        try:
            datasets = [xr.open_dataset(file) for file in variable_files]
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


def download_pressure_data(session, year, month, target_file, bbox, variables, plevels, time_steps):
    """
    Download ERA5 pressure level data using THREDDS Data Server.

    Args:
        session: Requests session with authentication
        year: Year to download
        month: Month to download
        target_file: Output file path
        bbox: Bounding box [latN, lonW, latS, lonE]
        variables: List of pressure level variables to download
        plevels: List of pressure levels to download
        time_steps: List of time steps to download

    Returns:
        Downloaded and merged data saved to target_file
    """
    print(f"Downloading pressure level data for {year}-{month:02d}...")

    # Create temporary directory for variable downloads
    temp_dir = f"{target_file}_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Convert time steps to hours for THREDDS
    hours = [int(ts.split(':')[0]) for ts in time_steps]

    # Calculate number of days in the month
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    days_in_month = (next_month - datetime(year, month, 1)).days

    # Download each variable
    variable_files = []
    for var in variables:
        # Create the file path
        var_file = os.path.join(temp_dir, f"{var}.nc")

        var_datasets = []
        for plev in plevels:
            # THREDDS URL for the variable
            # Corrected base URL for pressure level data - removed extra /e5.oper.an.pl and added level as subdirectory
            base_url = f"https://thredds.rda.ucar.edu/thredds/dodsC/g/ds633.0/e5.oper.an.pl/{year}/{var}/{plev}"

            # Construct the query parameters for the NCSS request
            params = {
                'var': var,
                'north': bbox[0],
                'west': bbox[1],
                'south': bbox[2],
                'east': bbox[3],
                'disableProjSubset': 'on',
                'vertCoord': plev, # Added vertical level subsetting
                'horizStride': 1,
                'time_start': f"{year}-{month:02d}-01T00:00:00Z",
                'time_end': f"{year}-{month:02d}-{days_in_month:02d}T23:00:00Z",
                'timeStride': 1,
                'time': [f"{h:02d}:00" for h in hours], # Added time subsetting
                'addLatLon': 'true',
                'accept': 'netcdf4' # Changed to netcdf4 for potential compatibility
            }

            # Build the request URL including the NetCDF Subset Service (NCSS) endpoint
            url = f"{base_url}.dodsC?"

            # Make the request
            response = session.get(url + urlencode(params), stream=True)

            if response.status_code == 200:
                plev_file = os.path.join(temp_dir, f"{var}_{plev}.nc")
                with open(plev_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                var_datasets.append(plev_file)
                print(f"  Downloaded {var} at {plev} hPa for {year}-{month:02d}")
            else:
                print(f"  Failed to download {var} at {plev} hPa for {year}-{month:02d}: {response.status_code}")

        # Combine all pressure levels for this variable
        if var_datasets:
            try:
                # Open all datasets for this variable
                plev_datasets = [xr.open_dataset(file) for file in var_datasets]

                # Combine datasets along level dimension
                combined_ds = xr.concat(plev_datasets, dim='level')
                combined_ds['level'] = np.array([int(x) for x in plevels]) # Ensure level coordinate is correct

                # Save combined dataset for this variable
                combined_ds.to_netcdf(var_file)

                # Close datasets
                for ds in plev_datasets:
                    ds.close()

                variable_files.append(var_file)
                print(f"  Combined pressure levels for {var} into {var_file}")

                # Clean up temporary files
                for file in var_datasets:
                    os.remove(file)
            except Exception as e:
                print(f"  Error combining pressure levels for {var}: {str(e)}")

    # Merge the downloaded variables into a single file
    if variable_files:
        try:
            # Open all datasets
            datasets = [xr.open_dataset(file) for file in variable_files]

            # Merge datasets
            merged_ds = xr.merge(datasets)

            # Save merged dataset
            merged_ds.to_netcdf(target_file)

            # Close datasets
            for ds in datasets:
                ds.close()

            print(f"Merged pressure level variables into {target_file}")

            # Clean up temporary files
            shutil.rmtree(temp_dir)

            return True
        except Exception as e:
            print(f"Error merging pressure level variables: {str(e)}")
            return False
    else:
        print(f"No pressure level variables downloaded for {year}-{month:02d}")
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
    start_date = "2022-01-01"
    end_date = "2022-01-31"
    output_dir = "./era5_data"

    # Geographic region (Europe)
    lat_north = 60.0
    lat_south = 35.0
    lon_east = 30.0
    lon_west = -10.0

    # Surface variables to download
    # surf_vars = ['2t', '2d', 'sp', 'tp']
    surf_vars = None

    # Pressure level variables and levels
    plev_vars = ['t', 'z', 'q']
    plevels = ['850', '500', '250']

    # Download the data
    files = retrieve_era5_ncar(
        start_date, end_date, output_dir,
        lat_north, lat_south, lon_east, lon_west,
        surf_vars, plev_vars, plevels,
        time_step='6H',
        num_threads=1,
        orcid_id="rnebot@gmail.com", orcid_password="Barr1ales."
    )

    print("Downloaded files:")
    for key, file_list in files.items():
        print(f"{key}:")
        for file in file_list:
            print(f"  {file}")