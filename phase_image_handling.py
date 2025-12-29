# phase_image_handling.py
"""
Handles image loading, interactive selection, KML creation, and diagnostics.
Everything up to "Diagnostic Check Passed. Proceeding to Parameter Configuration..."
"""

import os
import glob
import datetime
import numpy as np
from SAR_interactive_selector import run_interactive_selector, get_pixel_geo_coord
from phase_0_diagnostics import run_diagnostic_check
from kml_generator import create_dumb_kml
from data_loader import load_mlc_data
from Ext_Data import save_config


def get_box_corners_geo(r_start, r_end, c_start, c_end, model):
    """Helper to get 4 Lat/Lon corners for KML using Selector math"""
    p1 = get_pixel_geo_coord(r_start, c_start, model)  # UL
    p2 = get_pixel_geo_coord(r_start, c_end, model)    # UR
    p3 = get_pixel_geo_coord(r_end, c_end, model)      # LR
    p4 = get_pixel_geo_coord(r_end, c_start, model)    # LL
    return [p1, p2, p3, p4]


def run_image_selection_pipeline(file_paths, full_config, run_timestamp):
    """
    Main pipeline for image handling:
    1. Loads radar parameters
    2. Runs interactive selector
    3. Generates KML
    4. Runs diagnostic check
    5. Returns all necessary data for main processing
    
    Returns: (complex_data_full, radar_params, user_selection, center_row, central_col_idx)
    """
    from data_loader import parse_radar_parameters
    
    print(f"\n=== DOPPLER ANALYSIS PARAMETERS ===")
    print(f"Loaded from SAR metadata:")
    
    radar_params = parse_radar_parameters(file_paths['json_file'])
    if not radar_params:
        return None, None, None, None, None
    
    # --- DOPPLER ANALYSIS PARAMETER LOGGING ---
    doppler_params_available = False
    doppler_params_missing = []
    
    # Check and log each Doppler parameter
    if 'doppler_bandwidth_hz' in radar_params and radar_params['doppler_bandwidth_hz'] > 0:
        print(f"  ✓ Doppler bandwidth: {radar_params['doppler_bandwidth_hz']:.2f} Hz")
        doppler_params_available = True
    else:
        print(f"  ✗ Doppler bandwidth: MISSING")
        doppler_params_missing.append("doppler_bandwidth_hz")
    
    if 'doppler_centroid_initial_hz' in radar_params:
        print(f"  ✓ Doppler centroid: {radar_params['doppler_centroid_initial_hz']:.2f} Hz")
        doppler_params_available = True
    else:
        print(f"  ✗ Doppler centroid: MISSING (using 0 Hz)")
        doppler_params_missing.append("doppler_centroid_initial_hz")
    
    if 'effective_prf_hz' in radar_params and radar_params['effective_prf_hz'] > 0:
        print(f"  ✓ Effective PRF: {radar_params['effective_prf_hz']:.2f} Hz")
        doppler_params_available = True
    else:
        print(f"  ✗ Effective PRF: MISSING")
        doppler_params_missing.append("effective_prf_hz")
    
    if 'doppler_ambiguity_spacing_hz' in radar_params and radar_params['doppler_ambiguity_spacing_hz'] > 0:
        print(f"  ✓ Doppler ambiguity spacing: {radar_params['doppler_ambiguity_spacing_hz']:.2f} Hz")
        doppler_params_available = True
    else:
        print(f"  ✗ Doppler ambiguity spacing: MISSING")
        doppler_params_missing.append("doppler_ambiguity_spacing_hz")
    
    if 'max_unambiguous_doppler_hz' in radar_params and radar_params['max_unambiguous_doppler_hz'] > 0:
        print(f"  ✓ Max unambiguous Doppler: ±{radar_params['max_unambiguous_doppler_hz']:.2f} Hz")
        doppler_params_available = True
    else:
        print(f"  ✗ Max unambiguous Doppler: MISSING")
        doppler_params_missing.append("max_unambiguous_doppler_hz")
    
    # Check if we have time-varying PRF data
    if 'prf_values' in radar_params and radar_params['prf_values']:
        print(f"  ✓ Time-varying PRF: {len(radar_params['prf_values'])} samples")
        print(f"    Mean: {radar_params['prf_mean']:.2f} Hz, Std: {radar_params['prf_std']:.2f} Hz")
        print(f"    Range: [{radar_params['prf_min']:.2f}, {radar_params['prf_max']:.2f}] Hz")
        doppler_params_available = True
    else:
        print(f"  ✗ Time-varying PRF: MISSING (using static PRF)")
        doppler_params_missing.append("prf_values")
    
    # Summary of Doppler parameter availability
    if doppler_params_available:
        print(f"\n[DOPPLER] Some Doppler parameters available. Will use them for sub-aperture generation.")
        if doppler_params_missing:
            print(f"[DOPPLER WARNING] Missing parameters: {', '.join(doppler_params_missing)}")
            print(f"[DOPPLER] Will use defaults for missing parameters.")
    else:
        print(f"\n[DOPPLER WARNING] No Doppler parameters available!")
        print(f"[DOPPLER] Sub-aperture generation will use default frequency axis division.")
    
    # Store Doppler availability for later use
    radar_params['doppler_params_available'] = doppler_params_available
    
    # --- SELECTION & DIAGNOSTIC LOOP ---
    vis_data_cache = None
    user_selection = None
    
    while True:
        # Pass cache to skip reload if looping
        selection_result = run_interactive_selector(file_paths, full_config, preloaded_data=vis_data_cache)
        
        if not selection_result or selection_result[0] is None:
            print("Selection cancelled. Exiting.")
            return None, None, None, None, None

        user_selection, vis_data_cache, new_defaults = selection_result

        # --- UPDATED: SAVE DEFAULTS IMMEDIATELY ---
        print("\n--- Updating Configuration with New Defaults ---")
        if 'calibration' not in full_config:
            full_config['calibration'] = {}
        if 'user_parameters' not in full_config:
            full_config['user_parameters'] = {}
        if 'column_ranges' not in full_config:
            full_config['column_ranges'] = {}
        
        # Update specific sections
        full_config['calibration'].update(new_defaults['calibration'])
        full_config['user_parameters'].update(new_defaults['user_parameters'])
        full_config['column_ranges'].update(new_defaults['column_ranges'])
        
        # Also save geo center if available
        if user_selection.get('lat_center'):
            if 'geo_coords' not in full_config:
                full_config['geo_coords'] = {}
            full_config['geo_coords']['lat_start'] = str(user_selection['lat_center'])
            full_config['geo_coords']['lon_start'] = str(user_selection['lon_center'])

        save_config(full_config)

        # Update Radar Params with valid corners
        if 'corners' in user_selection:
            print("\n--- Updating Radar Metadata with Valid Interactive Map Coordinates ---")
            # corners from selector now contains the full model
            model_corners = user_selection['corners']
            radar_params.update({
                'lat_upper_left': model_corners['ul'][0],
                'lon_upper_left': model_corners['ul'][1],
                'lat_upper_right': model_corners['ur'][0],
                'lon_upper_right': model_corners['ur'][1],
                'lat_lower_left': model_corners['ll'][0],
                'lon_lower_left': model_corners['ll'][1],
                'lat_lower_right': model_corners['lr'][0],
                'lon_lower_right': model_corners['lr'][1]
            })

        center_row = user_selection['center_row']
        central_col_idx = user_selection['center_col']

        # --- KML GENERATION (USING SELECTOR MATH) - AUTOMATIC ---
        print("\n--- 📄 Generating KML analysis file ---")
        # The 'corners' object in user_selection is the full model dict including rows/cols
        calibrated_model = user_selection['corners']
        
        # We need pixel radius again for rows
        spacing_m = radar_params.get('azimuth_spacing_m', 1.0)
        extent_km = user_selection["analysis_extent_km"]
        extent_in_pixels = int(round((extent_km * 1000) / spacing_m))
        r_start = max(0, center_row - extent_in_pixels)
        r_end = min(radar_params['scene_rows'] - 1, center_row + extent_in_pixels)
        
        # Column bounds
        s_l, e_l = user_selection['start_left'], user_selection['end_left']
        s_r, e_r = user_selection['start_right'], user_selection['end_right']
        
        # Calculate Polygons (Left, Center, Right, Yellow)
        polys = {}
        
        # Yellow Box (Full Extent approx)
        polys['yellow'] = get_box_corners_geo(
            r_start, r_end, 
            max(0, central_col_idx - e_l), 
            min(radar_params['scene_cols'] - 1, central_col_idx + e_r), 
            calibrated_model
        )
        
        # Red (Left Strip)
        cs_l = max(0, central_col_idx - e_l)
        ce_l = max(0, central_col_idx - s_l)
        if ce_l >= cs_l:
            polys['red'] = get_box_corners_geo(r_start, r_end, cs_l, ce_l, calibrated_model)
        
        # Violet (Right Strip)
        cs_r = min(radar_params['scene_cols'] - 1, central_col_idx + s_r)
        ce_r = min(radar_params['scene_cols'] - 1, central_col_idx + e_r)
        if ce_r >= cs_r:
            polys['violet'] = get_box_corners_geo(r_start, r_end, cs_r, ce_r, calibrated_model)
        
        kml_filename = f"analysis_area_{run_timestamp}.kml"
        kml_path = os.path.join(os.getcwd(), kml_filename)
        create_dumb_kml(kml_path, polys)
        print(f"  ✓ KML saved to {kml_path}")

        # --- DIAGNOSTIC CHECK ---
        print("\n--- 🏥 PHASE 0: PRE-FLIGHT SIGNAL CHECK ---")
        print("Loading FULL resolution data for diagnostics (cannot be skipped)...")
        complex_data_full = load_mlc_data(file_paths['tiff_file'], radar_params)
        
        if complex_data_full is not None:
            should_proceed = run_diagnostic_check(complex_data_full, center_row, central_col_idx)
            if should_proceed:
                print("\n✅  Diagnostic Check Passed. Proceeding to Parameter Configuration...")
                return complex_data_full, radar_params, user_selection, center_row, central_col_idx
            else:
                print("\n⚠️  User rejected diagnostics. Returning to Map Selection...")
        else:
            print("Error loading data for diagnostics. Aborting.")
            return None, None, None, None, None