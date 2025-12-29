# Ext_Data.py

import configparser
import os
import glob
import re

CONFIG_FILE = 'config.ini'

def _create_default_config():
    """Creates a default config file if it doesn't exist."""
    config = configparser.ConfigParser()
    print(f"Configuration file '{CONFIG_FILE}' not found. Creating with default values.", flush=True)
    # Add sections to prevent errors on first run
    config['Filenames'] = {}
    config['GeoCoordinates'] = {
        'lat_start': '48.8566', # Default to Paris
        'lon_start': '2.3522',
        'lat_end': '48.86',
        'lon_end': '2.36'
    }
    config['user_parameters'] = {}
    config['column_ranges'] = {}
    config['calibration'] = {}  # ADDED: calibration section for storing calibration data
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)
    return config

def load_config():
    """Loads file paths and geo coordinates from the configuration file."""
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        config = _create_default_config()
    else:
        config.read(CONFIG_FILE)
    
    # FIX: Handle both section naming conventions (with and without underscore)
    # First check for sections with underscore (new convention)
    user_params_section = 'user_parameters'
    geo_coords_section = 'GeoCoordinates' 
    column_ranges_section = 'column_ranges'
    calibration_section = 'calibration'  # ADDED: calibration section name
    
    # If the new sections don't exist, check for old sections without underscore
    if not config.has_section(user_params_section) and config.has_section('user_parameters'):
        user_params_section = 'user_parameters'
    if not config.has_section(geo_coords_section) and config.has_section('geo_coords'):
        geo_coords_section = 'geo_coords'
    if not config.has_section(column_ranges_section) and config.has_section('column_ranges'):
        column_ranges_section = 'column_ranges'
    if not config.has_section(calibration_section) and config.has_section('calibration'):
        calibration_section = 'calibration'
    
    # Ensure all sections exist to prevent KeyErrors
    if not config.has_section('GeoCoordinates'): 
        config.add_section('GeoCoordinates')
    if not config.has_section('user_parameters'): 
        config.add_section('user_parameters')
    if not config.has_section('column_ranges'): 
        config.add_section('column_ranges')
    if not config.has_section('calibration'):  # ADDED: ensure calibration section exists
        config.add_section('calibration')
        
    filenames = dict(config['Filenames']) if config.has_section('Filenames') else {}
    
    # FIX: Handle geo coordinates with proper section name
    geo_coords = {}
    if config.has_section('GeoCoordinates'):
        geo_coords = {k: float(v) for k, v in config['GeoCoordinates'].items()}
    elif config.has_section('geo_coords'):
        geo_coords = {k: float(v) for k, v in config['geo_coords'].items()}
    
    # FIX: Handle user parameters with proper section name  
    user_parameters = {}
    if config.has_section('user_parameters'):
        user_parameters = dict(config['user_parameters'])
    elif config.has_section('user_parameters'):
        user_parameters = dict(config['user_parameters'])
    
    # FIX: Handle column ranges with proper section name
    column_ranges = {}
    if config.has_section('column_ranges'):
        column_ranges = {k: int(v) for k, v in config['column_ranges'].items()}
    elif config.has_section('column_ranges'):
        column_ranges = {k: int(v) for k, v in config['column_ranges'].items()}
    
    # ADDED: Handle calibration section
    calibration_params = {}
    if config.has_section('calibration'):
        calibration_params = dict(config['calibration'])
        # Convert numeric values to appropriate types
        for key, value in calibration_params.items():
            try:
                # Try to convert to float if it looks like a number
                if '.' in value:
                    calibration_params[key] = float(value)
                else:
                    calibration_params[key] = int(value)
            except (ValueError, AttributeError):
                # Keep as string if conversion fails
                pass

    full_config = {
        'filenames': filenames,
        'geo_coords': geo_coords,
        'user_parameters': user_parameters,
        'column_ranges': column_ranges,
        'calibration': calibration_params  # ADDED: include calibration data
    }
    return full_config, geo_coords

def save_config(config_dict):
    """Saves the provided configuration dictionary to the file."""
    config = configparser.ConfigParser()
    
    # FIRST: Read the existing config file to preserve all current values
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
    
    # SECOND: Update ONLY the sections and keys provided in config_dict
    for section, values in config_dict.items():
        # FIX: Normalize section names to use consistent naming
        normalized_section = section
        if section == 'user_parameters':
            normalized_section = 'user_parameters'
        elif section == 'geo_coords':
            normalized_section = 'GeoCoordinates'
        elif section == 'column_ranges':
            normalized_section = 'column_ranges'
        elif section == 'calibration':
            normalized_section = 'calibration'  # ADDED: handle calibration section
        
        # Ensure the section exists
        if not config.has_section(normalized_section):
            config.add_section(normalized_section)
        
        # Update each key-value pair in the section
        for key, value in values.items():
            config[normalized_section][key] = str(value)

    # THIRD: Write the updated config back to file
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)
    print(f"Configuration saved to '{CONFIG_FILE}'.", flush=True)

def get_external_data_paths():
    """
    Main function to auto-detect Capella datasets and let the user choose.
    """
    print("\n--- 📂 0. Discovering Datasets ---", flush=True)
    
    # Search for Capella SLC GeoTIFF files
    tiff_files = glob.glob('*_SLC_*.tif')
    valid_datasets = []

    for tiff_file in tiff_files:
        # Expect a corresponding JSON file
        base_name = os.path.splitext(tiff_file)[0]
        json_file = f"{base_name}.json"
        if os.path.exists(json_file):
            valid_datasets.append({
                'json_file': json_file,
                'tiff_file': tiff_file,
                'name': base_name
            })

    if not valid_datasets:
        print("Error: No valid Capella SLC datasets (.tif + .json file pairs) found in this folder.")
        print("Please place your Capella SLC .tif and .json files in the same directory as the script.")
        return None

    print("Found the following Capella SLC datasets:")
    for i, dataset in enumerate(valid_datasets):
        print(f"  [{i+1}]: {dataset['name']}")

    choice = -1
    while True:
        try:
            choice_str = input(f"Which dataset do you want to process? (1-{len(valid_datasets)}): ")
            choice = int(choice_str) - 1
            if 0 <= choice < len(valid_datasets):
                break
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    selected_dataset = valid_datasets[choice]
    print(f"Selected dataset: {selected_dataset['name']}", flush=True)
    
    # The function now returns the paths for the chosen Capella dataset
    return selected_dataset