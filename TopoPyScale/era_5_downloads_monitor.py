import gspread
from google.oauth2.service_account import Credentials
import os
import glob
import time
import datetime
import logging

# --- Configuration ---

# 1. Google Sheets Details
SPREADSHEET_ID = '1gXITbRRluspFw6yy6O-kC5UBkTs0isFUegw54QFYW6c' # Extract from the URL
SHEET_NAME = 'Tasks'
# 2. Path to your downloaded Service Account JSON key file
SERVICE_ACCOUNT_FILE = '/mnt/datos/rnebot/genesis/TopoPyScale/genesis-454914-d3b7ba1987a8.json' # <--- CHANGE THIS
# 3. Update Interval (in seconds)
UPDATE_INTERVAL_SECONDS = 600 # e.g., 60 seconds = 1 minute

# 4. List of tasks to monitor
#    directory: Absolute path or relative path from where the script runs
#    file_mask: Glob pattern (e.g., '*.txt', 'data_*.csv')
#    total_number: The denominator for the ratio calculation
#    target_row: The row number in the Google Sheet to update
surf_mask = 'SURF_??????.nc'
plev_mask = 'PLEV_??????.nc'
n_files = 361

TASKS = [
    # {
    #     'directory': '/mnt/datos/rnebot/genesis/ERA5_Downscaling_LasPalmas/inputs/climate/',
    #     'file_mask': surf_mask,
    #     'total_number': n_files,
    #     'target_row': 19
    # },
    {
        'directory': '/mnt/datos/rnebot/genesis/ERA5_Downscaling_LasPalmas/inputs/climate/',
        'file_mask': plev_mask,
        'total_number': n_files,
        'target_row': 20
    },
    # {
    #     'directory': '/mnt/datos/rnebot/genesis/ERA5_Downscaling_Tenerife/inputs/climate/',
    #     'file_mask': surf_mask,
    #     'total_number': n_files,
    #     'target_row': 21
    # },
    {
        'directory': '/mnt/datos/rnebot/genesis/ERA5_Downscaling_Tenerife/inputs/climate/',
        'file_mask': plev_mask,
        'total_number': 361,
        'target_row': 22
    },
    {
        'directory': '/mnt/datos/rnebot/genesis/DownscalingTopoPyScale/ERA5_Downscaling_Canarias/inputs/climate/',
        'file_mask': surf_mask,
        'total_number': n_files,
        'target_row': 13
    },
    {
        'directory': '/mnt/datos/rnebot/genesis/DownscalingTopoPyScale/ERA5_Downscaling_Canarias/inputs/climate/',
        'file_mask': plev_mask,
        'total_number': n_files,
        'target_row': 14
    },
    # {
    #     'directory': '/mnt/datos/rnebot/genesis/DownscalingTopoPyScale/ERA5_Downscaling_Madeira/inputs/climate/',
    #     'file_mask': surf_mask,
    #     'total_number': n_files,
    #     'target_row': 11
    # },
    {
        'directory': '/mnt/datos/rnebot/genesis/DownscalingTopoPyScale/ERA5_Downscaling_Madeira/inputs/climate/',
        'file_mask': plev_mask,
        'total_number': n_files,
        'target_row': 12
    },
    {
        'directory': '/mnt/datos/rnebot/genesis/DownscalingTopoPyScale/ERA5_Downscaling_Azores/inputs/climate/',
        'file_mask': surf_mask,
        'total_number': n_files,
        'target_row': 15
    },
    {
        'directory': '/mnt/datos/rnebot/genesis/DownscalingTopoPyScale/ERA5_Downscaling_Azores/inputs/climate/',
        'file_mask': plev_mask,
        'total_number': n_files,
        'target_row': 16
    },
    {
        'directory': '/mnt/datos/rnebot/genesis/DownscalingTopoPyScale/ERA5_Downscaling_CaboVerde/inputs/climate/',
        'file_mask': surf_mask,
        'total_number': n_files,
        'target_row': 17
    },
    {
        'directory': '/mnt/datos/rnebot/genesis/DownscalingTopoPyScale/ERA5_Downscaling_CaboVerde/inputs/climate/',
        'file_mask': plev_mask,
        'total_number': n_files,
        'target_row': 18
    }
]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Google Sheets Authentication ---
def get_gspread_client():
    """Authenticates with Google Sheets API using service account."""
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets',
                  'https://www.googleapis.com/auth/drive.file'] # Required scopes
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        logging.info("Successfully authenticated with Google Sheets API.")
        return client
    except FileNotFoundError:
        logging.error(f"Service account key file not found at: {SERVICE_ACCOUNT_FILE}")
        return None
    except Exception as e:
        logging.error(f"Error during Google Sheets authentication: {e}")
        return None

# --- Main Monitoring Loop ---
def main():
    gc = get_gspread_client()
    if not gc:
        logging.error("Exiting due to authentication failure.")
        return
    # --- Debugging ---
    print("-" * 20)
    print(f"DEBUG: Type of gc object is: {type(gc)}")
    print(f"DEBUG: Does gc have 'open_by_id'? {'open_by_id' in dir(gc)}")
    # Optional: Print all attributes if the above is False
    print(f"DEBUG: Attributes of gc: {dir(gc)}")
    print("-" * 20)
    # --- End Debugging ---
    try:
        spreadsheet = gc.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        logging.info(f"Opened spreadsheet '{spreadsheet.title}' and worksheet '{SHEET_NAME}'.")
    except gspread.exceptions.SpreadsheetNotFound:
        logging.error(f"Spreadsheet with ID '{SPREADSHEET_ID}' not found or not accessible.")
        return
    except gspread.exceptions.WorksheetNotFound:
        logging.error(f"Worksheet named '{SHEET_NAME}' not found in the spreadsheet.")
        return
    except Exception as e:
        logging.error(f"Error opening spreadsheet/worksheet: {e}")
        return

    while True:
        logging.info("--- Starting monitoring cycle ---")
        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for task in TASKS:
            directory = task['directory']
            file_mask = task['file_mask']
            total_number = task['total_number']
            target_row = task['target_row']

            logging.info(f"Processing task for directory: {directory}, mask: {file_mask}, row: {target_row}")

            try:
                # Construct the full search path
                search_path = os.path.join(directory, file_mask)

                # Check if directory exists
                if not os.path.isdir(directory):
                    logging.warning(f"Directory not found: {directory}. Skipping task for row {target_row}.")
                    # Optionally update sheet with an error message
                    # worksheet.update_cell(target_row, 2, "Error: Dir Not Found")
                    # worksheet.update_cell(target_row, 3, current_timestamp)
                    continue

                # Find and count files
                matching_files = glob.glob(search_path)
                file_count = len(matching_files)
                logging.info(f"Found {file_count} files matching '{file_mask}' in '{directory}'.")

                # Calculate ratio
                completion_ratio = 0.0
                if total_number > 0:
                    completion_ratio = file_count / total_number
                else:
                    logging.warning(f"Task for row {target_row} has total_number=0. Setting ratio to 0.")

                # Update Google Sheet
                try:
                    # Update column B (Ratio) - Column B is index 2
                    worksheet.update_cell(target_row, 2, completion_ratio)
                    # Update column C (Timestamp) - Column C is index 3
                    worksheet.update_cell(target_row, 3, current_timestamp)
                    logging.info(f"Successfully updated row {target_row} with ratio {completion_ratio:.4f} and timestamp.")

                except gspread.exceptions.APIError as api_err:
                    logging.error(f"Google Sheets API Error updating row {target_row}: {api_err}")
                    # Implement retry logic here if needed
                except Exception as update_err:
                    logging.error(f"Unexpected error updating row {target_row}: {update_err}")

            except FileNotFoundError:
                 logging.warning(f"Directory not found during glob: {directory}. Skipping task for row {target_row}.")
            except PermissionError:
                logging.warning(f"Permission denied accessing directory: {directory}. Skipping task for row {target_row}.")
                # Optionally update sheet with an error message
                # worksheet.update_cell(target_row, 2, "Error: Permission Denied")
                # worksheet.update_cell(target_row, 3, current_timestamp)
            except Exception as task_err:
                logging.error(f"Error processing task for row {target_row}: {task_err}")

        logging.info(f"--- Cycle complete. Waiting for {UPDATE_INTERVAL_SECONDS} seconds. ---")
        time.sleep(UPDATE_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()