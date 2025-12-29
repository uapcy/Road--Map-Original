# data_loader.py - FIXED FOR CAPELLA JSON STRUCTURE WITH DOPPLER PARAMETERS

import numpy as np
import json
import os
import rasterio
import re

def parse_radar_parameters(json_path):
    """
    Parses parameters. Returns valid dict even if keys are missing (Safe Mode).
    FIXED: Extracts Doppler parameters for proper 96-look analysis.
    Now checks for extended JSON file with time_varying_parameters.
    """
    params = {}
    
    # Track what we found and what's missing
    loaded_params = []
    missing_params = []
    
    # Default fallback values (Safe X-band defaults)
    defaults = {
        'scene_rows': 0, 'scene_cols': 0,
        'center_frequency_hz': 9.65e9, 
        'bandwidth_hz': 500e6,
        'prf_hz': 3000.0,
        'incidence_angle_rad': 0.78, # ~45 deg
        'slant_range_m': 600000.0,
        'azimuth_spacing_m': 0.5,
        'range_spacing_m': 0.5,
        'ground_range_spacing_m': 0.5,  # Ground range spacing
        'image_length_m': 5000.0,  # Image length in meters
        'image_width_m': 5000.0,   # Image width in meters
        'centroid_lat': 48.857856955,  # Centroid from JSON
        'centroid_lon': 2.294476075,   # Centroid from JSON
        'lat_upper_left': 0.0, 'lon_upper_left': 0.0,
        'lat_lower_left': 0.0, 'lon_lower_left': 0.0,
        
        # NEW: Doppler parameter defaults
        'doppler_bandwidth_hz': 100000.0,  # Default Doppler bandwidth
        'doppler_centroid_polynomial': None,
        'effective_prf_hz': 3000.0,
        'doppler_ambiguity_spacing_hz': 3000.0,
        'max_unambiguous_doppler_hz': 1500.0,
        'doppler_bandwidth_per_look_hz': 1041.67,  # For 96 looks
        'prf_values': [],  # List of actual PRF values if available
        'prf_mean': 3000.0,
        'prf_std': 0.0,
        'prf_min': 3000.0,
        'prf_max': 3000.0
    }
    
    # Check for extended JSON file first
    base_path = os.path.splitext(json_path)[0]
    extended_json_path = f"{base_path}_extended.json"
    json_files_to_try = []
    
    if os.path.exists(extended_json_path):
        json_files_to_try.append(extended_json_path)
        print(f"[INFO] Found extended JSON file: {os.path.basename(extended_json_path)}")
    else:
        print(f"[INFO] Extended JSON file not found: {os.path.basename(extended_json_path)}")
    
    # Always try the original JSON file
    if os.path.exists(json_path):
        json_files_to_try.append(json_path)
        print(f"[INFO] Found original JSON file: {os.path.basename(json_path)}")
    else:
        print(f"[ERROR] Original JSON file not found: {json_path}")
        missing_params.append("JSON_FILE_NOT_FOUND")
        return defaults
    
    # Initialize metadata dictionaries
    meta_extended = None
    meta_original = None
    
    # Load JSON files
    for json_file in json_files_to_try:
        try:
            with open(json_file, 'r') as f:
                meta = json.load(f)
                
            if '_extended' in json_file:
                meta_extended = meta
                print(f"[INFO] Successfully loaded extended JSON metadata")
            else:
                meta_original = meta
                print(f"[INFO] Successfully loaded original JSON metadata")
                
        except Exception as e:
            print(f"[WARNING] Error loading JSON file {json_file}: {e}")
    
    # Use extended metadata if available, otherwise use original
    meta = meta_extended if meta_extended is not None else meta_original
    
    if meta is None:
        print(f"[ERROR] Could not load any JSON metadata")
        missing_params.append("JSON_METADATA_LOAD_FAILED")
        return defaults
    
    print(f"\n[INFO] === DOPPLER PARAMETER EXTRACTION ===")
    
    try:
        # Get scene dimensions - THEY ARE AT collect.image.rows/.columns
        if 'collect' in meta and 'image' in meta['collect']:
            image_meta = meta['collect']['image']
            params['scene_rows'] = image_meta.get('rows', 0)
            params['scene_cols'] = image_meta.get('columns', 0)
            loaded_params.append(f"scene_dimensions({params['scene_rows']}x{params['scene_cols']})")
            print(f"[LOADED] Scene dimensions: {params['scene_rows']} rows x {params['scene_cols']} cols")
        else:
            missing_params.append("scene_dimensions")
    except Exception as e:
        print(f"[DEBUG] Error reading scene dimensions: {e}")
        missing_params.append("scene_dimensions")
    
    try:
        # Get radar parameters
        if 'collect' in meta and 'radar' in meta['collect']:
            radar = meta['collect']['radar']
            params['center_frequency_hz'] = float(radar.get('center_frequency', defaults['center_frequency_hz']))
            params['bandwidth_hz'] = float(radar.get('bandwidth', defaults['bandwidth_hz']))
            
            loaded_params.append(f"center_frequency({params['center_frequency_hz']:.3e}Hz)")
            loaded_params.append(f"bandwidth({params['bandwidth_hz']:.3e}Hz)")
            
            print(f"[LOADED] Radar: freq={params['center_frequency_hz']:.3e} Hz, bw={params['bandwidth_hz']:.3e} Hz")
        else:
            missing_params.append("radar_parameters")
    except Exception as e:
        print(f"[DEBUG] Error reading radar params: {e}")
        missing_params.append("radar_parameters")
    
    # ====== NEW: EXTRACT DOPPLER PARAMETERS ======
    try:
        if 'collect' in meta and 'image' in meta['collect']:
            image_meta = meta['collect']['image']
            
            # 1. Doppler bandwidth
            if 'processed_azimuth_bandwidth' in image_meta:
                params['doppler_bandwidth_hz'] = float(image_meta['processed_azimuth_bandwidth'])
                loaded_params.append(f"doppler_bandwidth({params['doppler_bandwidth_hz']:.2f}Hz)")
                print(f"[LOADED] Doppler bandwidth: {params['doppler_bandwidth_hz']:.2f} Hz")
            else:
                missing_params.append("doppler_bandwidth")
                params['doppler_bandwidth_hz'] = defaults['doppler_bandwidth_hz']
                print(f"[MISSING] Doppler bandwidth - using default: {params['doppler_bandwidth_hz']:.2f} Hz")
            
            # 2. Doppler centroid polynomial
            if 'frequency_doppler_centroid_polynomial' in image_meta:
                params['doppler_centroid_polynomial'] = image_meta['frequency_doppler_centroid_polynomial']
                loaded_params.append("doppler_centroid_polynomial")
                print(f"[LOADED] Doppler centroid polynomial available")
                
                # Extract first coefficient as initial Doppler centroid
                try:
                    coeffs = params['doppler_centroid_polynomial']['coefficients']
                    if coeffs and len(coeffs) > 0 and len(coeffs[0]) > 0:
                        params['doppler_centroid_initial_hz'] = float(coeffs[0][0])
                        print(f"[LOADED] Initial Doppler centroid: {params['doppler_centroid_initial_hz']:.2f} Hz")
                except:
                    params['doppler_centroid_initial_hz'] = 0.0
            else:
                missing_params.append("doppler_centroid_polynomial")
                params['doppler_centroid_polynomial'] = None
                params['doppler_centroid_initial_hz'] = 0.0
                print(f"[MISSING] Doppler centroid polynomial")
            
            # 3. Range bandwidth (for completeness)
            if 'processed_range_bandwidth' in image_meta:
                params['range_bandwidth_hz'] = float(image_meta['processed_range_bandwidth'])
                loaded_params.append(f"range_bandwidth({params['range_bandwidth_hz']:.2f}Hz)")
                print(f"[LOADED] Range bandwidth: {params['range_bandwidth_hz']:.2f} Hz")
            else:
                missing_params.append("range_bandwidth")
                
    except Exception as e:
        print(f"[DEBUG] Error reading Doppler parameters: {e}")
        missing_params.append("doppler_parameters")
    
    # ====== NEW: EXTRACT TIME-VARYING PRF PARAMETERS ======
    try:
        # Check if we have time_varying_parameters (likely in extended JSON)
        time_varying_prfs = []
        
        if 'collect' in meta and 'radar' in meta['collect']:
            radar = meta['collect']['radar']
            
            if 'time_varying_parameters' in radar and radar['time_varying_parameters']:
                for tvp in radar['time_varying_parameters']:
                    if 'prf' in tvp:
                        time_varying_prfs.append(float(tvp['prf']))
                
                if time_varying_prfs:
                    params['prf_values'] = time_varying_prfs
                    params['prf_mean'] = np.mean(time_varying_prfs)
                    params['prf_std'] = np.std(time_varying_prfs)
                    params['prf_min'] = np.min(time_varying_prfs)
                    params['prf_max'] = np.max(time_varying_prfs)
                    params['effective_prf_hz'] = params['prf_mean']
                    
                    loaded_params.append(f"time_varying_prf({len(time_varying_prfs)} samples)")
                    print(f"[LOADED] Time-varying PRF: {len(time_varying_prfs)} samples")
                    print(f"[LOADED] PRF stats: mean={params['prf_mean']:.2f} Hz, "
                          f"std={params['prf_std']:.2f} Hz, "
                          f"range=[{params['prf_min']:.2f}, {params['prf_max']:.2f}] Hz")
                    
            # Also check for simple PRF array
            elif 'prf' in radar and isinstance(radar['prf'], list) and len(radar['prf']) > 0:
                prf_value = float(radar['prf'][0])
                params['effective_prf_hz'] = prf_value
                params['prf_mean'] = prf_value
                loaded_params.append(f"prf_array({prf_value:.2f}Hz)")
                print(f"[LOADED] PRF from array: {prf_value:.2f} Hz")
            else:
                missing_params.append("time_varying_prf")
                params['effective_prf_hz'] = defaults['effective_prf_hz']
                params['prf_mean'] = defaults['prf_mean']
                print(f"[MISSING] Time-varying PRF - using default: {params['effective_prf_hz']:.2f} Hz")
                
    except Exception as e:
        print(f"[DEBUG] Error reading time-varying PRF: {e}")
        missing_params.append("time_varying_prf")
        params['effective_prf_hz'] = defaults['effective_prf_hz']
        params['prf_mean'] = defaults['prf_mean']
    
    # ====== CALCULATE DOPPLER ANALYSIS PARAMETERS ======
    try:
        # Doppler ambiguity spacing = effective PRF
        params['doppler_ambiguity_spacing_hz'] = params['effective_prf_hz']
        loaded_params.append(f"doppler_ambiguity_spacing({params['doppler_ambiguity_spacing_hz']:.2f}Hz)")
        
        # Maximum unambiguous Doppler frequency = PRF/2
        params['max_unambiguous_doppler_hz'] = params['effective_prf_hz'] / 2
        loaded_params.append(f"max_unambiguous_doppler(±{params['max_unambiguous_doppler_hz']:.1f}Hz)")
        
        # Calculate bandwidth per look for 96 looks
        params['doppler_bandwidth_per_look_hz'] = params['doppler_bandwidth_hz'] / 96
        loaded_params.append(f"doppler_bandwidth_per_look({params['doppler_bandwidth_per_look_hz']:.2f}Hz)")
        
        print(f"[CALCULATED] Doppler parameters for 96-look analysis:")
        print(f"  - Doppler ambiguity spacing: {params['doppler_ambiguity_spacing_hz']:.2f} Hz")
        print(f"  - Max unambiguous Doppler: ±{params['max_unambiguous_doppler_hz']:.1f} Hz")
        print(f"  - Bandwidth per look (96 looks): {params['doppler_bandwidth_per_look_hz']:.2f} Hz")
        
        # Calculate overlap parameters
        if 'OVERLAP_FACTOR' in params.get('user_parameters', {}):
            overlap = float(params['user_parameters']['OVERLAP_FACTOR'])
        else:
            overlap = 0.85  # Default overlap factor
            
        params['doppler_look_overlap_hz'] = params['doppler_bandwidth_per_look_hz'] * overlap
        params['doppler_look_effective_bandwidth_hz'] = params['doppler_bandwidth_per_look_hz'] * (1 - overlap)
        
        print(f"  - Look overlap ({overlap*100:.0f}%): {params['doppler_look_overlap_hz']:.2f} Hz")
        print(f"  - Effective look bandwidth: {params['doppler_look_effective_bandwidth_hz']:.2f} Hz")
        
    except Exception as e:
        print(f"[DEBUG] Error calculating Doppler parameters: {e}")
        missing_params.append("doppler_calculations")
    
    # ====== EXISTING CODE (preserved) ======
    try:
        # Get incidence angle and slant range
        if 'collect' in meta and 'image' in meta['collect'] and 'center_pixel' in meta['collect']['image']:
            center_pixel = meta['collect']['image']['center_pixel']
            incidence_angle_deg = float(center_pixel.get('incidence_angle', 45.0))
            params['incidence_angle_rad'] = np.radians(incidence_angle_deg)
            params['incidence_angle_deg'] = incidence_angle_deg  # Store degrees too
            
            # Slant range might not be directly in JSON, use target position to calculate or use default
            params['slant_range_m'] = defaults['slant_range_m']
            loaded_params.append(f"incidence_angle({incidence_angle_deg:.1f}°)")
            print(f"[LOADED] Center pixel: incidence={incidence_angle_deg:.1f}°")
        else:
            missing_params.append("center_pixel_parameters")
    except Exception as e:
        print(f"[DEBUG] Error reading center pixel: {e}")
        missing_params.append("center_pixel_parameters")
    
    try: 
        # Get spacing values - CAPELLA HAS BOTH DISPLAY AND FULL-RESOLUTION SPACING
        if 'collect' in meta and 'image' in meta['collect']:
            image_meta = meta['collect']['image']
            
            # 1. DISPLAY SPACING (for the actual displayed image)
            # These are the pixel spacings for the 5km x 5km image
            display_row_spacing = image_meta.get('pixel_spacing_row', None)
            display_col_spacing = image_meta.get('pixel_spacing_column', None)
            
            if display_row_spacing is not None:
                params['display_ground_range_spacing_m'] = float(display_row_spacing)
                loaded_params.append(f"display_ground_spacing({params['display_ground_range_spacing_m']:.6f}m)")
                print(f"[LOADED] Display row spacing (ground): {params['display_ground_range_spacing_m']:.6f} m")
            else:
                missing_params.append("display_row_spacing")
            
            if display_col_spacing is not None:
                params['display_azimuth_spacing_m'] = float(display_col_spacing)
                loaded_params.append(f"display_azimuth_spacing({params['display_azimuth_spacing_m']:.6f}m)")
                print(f"[LOADED] Display column spacing (azimuth): {params['display_azimuth_spacing_m']:.6f} m")
            else:
                missing_params.append("display_col_spacing")
            
            # 2. FULL-RESOLUTION SPACING (from image_geometry)
            if 'image_geometry' in image_meta:
                geom = image_meta['image_geometry']
                
                # Azimuth spacing (column spacing)
                if 'col_sample_spacing' in geom:
                    params['azimuth_spacing_m'] = float(geom['col_sample_spacing'])
                    loaded_params.append(f"fullres_azimuth_spacing({params['azimuth_spacing_m']:.6f}m)")
                    print(f"[LOADED] Full-res azimuth spacing: {params['azimuth_spacing_m']:.6f} m")
                else:
                    missing_params.append("col_sample_spacing")
                
                # Range spacing (row spacing) - THIS IS SLANT RANGE!
                if 'row_sample_spacing' in geom:
                    slant_range_spacing_m = float(geom['row_sample_spacing'])
                    params['range_spacing_m'] = slant_range_spacing_m  # SLANT range spacing
                    loaded_params.append(f"fullres_range_spacing({slant_range_spacing_m:.6f}m)")
                    print(f"[LOADED] Full-res range spacing (SLANT): {slant_range_spacing_m:.6f} m")
                    
                    # Calculate ground range spacing using incidence angle
                    if 'incidence_angle_rad' in params:
                        sin_incidence = np.sin(params['incidence_angle_rad'])
                        if sin_incidence > 0.01:  # Avoid division by zero
                            params['ground_range_spacing_m'] = slant_range_spacing_m / sin_incidence
                            loaded_params.append(f"calculated_ground_spacing({params['ground_range_spacing_m']:.6f}m)")
                            print(f"[LOADED] Calculated ground range spacing: {params['ground_range_spacing_m']:.6f} m (slant: {slant_range_spacing_m:.6f} m, sin(inc): {sin_incidence:.3f})")
                        else:
                            params['ground_range_spacing_m'] = slant_range_spacing_m
                            loaded_params.append(f"ground_spacing_fallback({slant_range_spacing_m:.6f}m)")
                    else:
                        params['ground_range_spacing_m'] = slant_range_spacing_m
                        loaded_params.append(f"ground_spacing_no_inc({slant_range_spacing_m:.6f}m)")
                else:
                    missing_params.append("row_sample_spacing")
            
            print(f"[INFO] Spacing summary:")
            print(f"  Display: row={params.get('display_ground_range_spacing_m', 'N/A'):.6f}m, col={params.get('display_azimuth_spacing_m', 'N/A'):.6f}m")
            print(f"  Full-res: azimuth={params.get('azimuth_spacing_m', 'N/A'):.6f}m, slant_range={params.get('range_spacing_m', 'N/A'):.6f}m, ground_range={params.get('ground_range_spacing_m', 'N/A'):.6f}m")
            
        else:
            missing_params.append("image_metadata")
    except Exception as e:
        print(f"[DEBUG] Error reading spacing from JSON: {e}")
        missing_params.append("spacing_parameters")
        pass

    # Get image dimensions
    try:
        if 'collect' in meta and 'image' in meta['collect']:
            image_meta = meta['collect']['image']
            params['image_length_m'] = float(image_meta.get('length', defaults['image_length_m']))
            params['image_width_m'] = float(image_meta.get('width', defaults['image_width_m']))
            loaded_params.append(f"image_dimensions({params['image_length_m']:.2f}x{params['image_width_m']:.2f}m)")
            print(f"[LOADED] Image dimensions: {params['image_length_m']:.2f} m x {params['image_width_m']:.2f} m")
        else:
            missing_params.append("image_dimensions")
    except Exception as e:
        print(f"[DEBUG] Error parsing image dimensions: {e}")
        missing_params.append("image_dimensions")
        pass
        
    # Try to parse centroid from JSON
    try:
        # Convert entire JSON to string for regex search
        json_str = json.dumps(meta)
        
        # Search for centroid pattern: "Lon2.294476075Lat48.857856955"
        centroid_pattern = r'Lon([\d\.]+)Lat([\d\.]+)'
        match = re.search(centroid_pattern, json_str)
        if match:
            params['centroid_lon'] = float(match.group(1))
            params['centroid_lat'] = float(match.group(2))
            loaded_params.append(f"centroid({params['centroid_lat']:.9f},{params['centroid_lon']:.9f})")
            print(f"[LOADED] Found centroid in JSON: {params['centroid_lat']:.9f}, {params['centroid_lon']:.9f}")
        else:
            missing_params.append("centroid")
            print(f"[MISSING] Centroid not found in JSON")
                
    except Exception as e:
        print(f"[DEBUG] Error parsing centroid: {e}")
        missing_params.append("centroid")
        pass

    # Merge with defaults for any missing keys
    for k, v in defaults.items():
        if k not in params:
            params[k] = v
    
    # Derived parameters
    params['wavelength_m'] = 299792458.0 / params['center_frequency_hz']
    loaded_params.append(f"wavelength({params['wavelength_m']:.4f}m)")

    # Verify calculations
    if 'scene_rows' in params and params['scene_rows'] > 0 and 'image_length_m' in params:
        calculated_display_spacing = params['image_length_m'] / params['scene_rows']
        print(f"[VERIFY] Calculated display spacing verification: {calculated_display_spacing:.6f} m = {params['image_length_m']:.2f}m / {params['scene_rows']} rows")
        
        if 'display_ground_range_spacing_m' not in params:
            params['display_ground_range_spacing_m'] = calculated_display_spacing
            loaded_params.append(f"calculated_display_ground_spacing({calculated_display_spacing:.6f}m)")
            print(f"[CALCULATED] Set display_ground_range_spacing_m = {params['display_ground_range_spacing_m']:.6f} m")
    
    if 'scene_cols' in params and params['scene_cols'] > 0 and 'image_width_m' in params:
        calculated_azimuth_spacing = params['image_width_m'] / params['scene_cols']
        print(f"[VERIFY] Calculated azimuth spacing verification: {calculated_azimuth_spacing:.6f} m = {params['image_width_m']:.2f}m / {params['scene_cols']} cols")
        
        if 'display_azimuth_spacing_m' not in params:
            params['display_azimuth_spacing_m'] = calculated_azimuth_spacing
            loaded_params.append(f"calculated_display_azimuth_spacing({calculated_azimuth_spacing:.6f}m)")
            print(f"[CALCULATED] Set display_azimuth_spacing_m = {params['display_azimuth_spacing_m']:.6f} m")
    
    # ====== FINAL SUMMARY ======
    print(f"\n[INFO] === PARAMETER LOADING SUMMARY ===")
    print(f"[INFO] LOADED parameters ({len(loaded_params)}):")
    for i, param in enumerate(sorted(loaded_params)[:20]):  # Show first 20
        print(f"  {i+1:2d}. {param}")
    if len(loaded_params) > 20:
        print(f"  ... and {len(loaded_params)-20} more")
    
    if missing_params:
        print(f"\n[INFO] MISSING parameters ({len(missing_params)}):")
        for i, param in enumerate(sorted(missing_params)[:10]):  # Show first 10
            print(f"  {i+1:2d}. {param}")
        if len(missing_params) > 10:
            print(f"  ... and {len(missing_params)-10} more")
    else:
        print(f"\n[INFO] No missing parameters - all expected data loaded successfully!")
    
    print(f"\n[INFO] === CRITICAL DOPPLER PARAMETERS FOR 96-LOOK ANALYSIS ===")
    print(f"  • Doppler bandwidth: {params.get('doppler_bandwidth_hz', 0):.2f} Hz")
    print(f"  • Bandwidth per look (96 looks): {params.get('doppler_bandwidth_per_look_hz', 0):.2f} Hz")
    print(f"  • Effective PRF: {params.get('effective_prf_hz', 0):.2f} Hz")
    print(f"  • Max unambiguous Doppler: ±{params.get('max_unambiguous_doppler_hz', 0):.1f} Hz")
    if 'doppler_centroid_polynomial' in params and params['doppler_centroid_polynomial']:
        print(f"  • Doppler centroid polynomial: AVAILABLE")
    else:
        print(f"  • Doppler centroid polynomial: MISSING (using 0 Hz)")
    
    if 'prf_values' in params and params['prf_values']:
        print(f"  • Time-varying PRF samples: {len(params['prf_values'])}")
        print(f"  • PRF range: [{params.get('prf_min', 0):.2f}, {params.get('prf_max', 0):.2f}] Hz")
    
    print("-" * 80)
    
    return params

def load_mlc_data(tiff_file, radar_params):
    if not os.path.exists(tiff_file): 
        print(f"Error: TIFF file not found: {tiff_file}")
        return None
    try:
        with rasterio.open(tiff_file) as src:
            data = src.read(1)
            # Update radar_params with actual dimensions from TIFF
            radar_params['scene_rows'] = data.shape[0]
            radar_params['scene_cols'] = data.shape[1]
            print(f"[INFO] Loaded TIFF: {data.shape[0]} rows x {data.shape[1]} cols")
            print(f"[INFO] TIFF data type: {data.dtype}, range: [{np.min(data):.3f}, {np.max(data):.3f}]")
            
            # Recalculate display spacing based on actual TIFF dimensions
            if 'image_length_m' in radar_params and radar_params['image_length_m'] > 0:
                radar_params['display_ground_range_spacing_m'] = radar_params['image_length_m'] / radar_params['scene_rows']
                print(f"[INFO] Recalculated display ground spacing from TIFF: {radar_params['display_ground_range_spacing_m']:.6f} m")
            
            if 'image_width_m' in radar_params and radar_params['image_width_m'] > 0:
                radar_params['display_azimuth_spacing_m'] = radar_params['image_width_m'] / radar_params['scene_cols']
                print(f"[INFO] Recalculated display azimuth spacing from TIFF: {radar_params['display_azimuth_spacing_m']:.6f} m")
            
            return data
    except Exception as e:
        print(f"Error loading TIFF {tiff_file}: {e}")
        return None

def get_pixel_geo_coord(row, col, params):
    # Simplified for stability
    lat = params.get('lat_upper_left', 0)
    lon = params.get('lon_upper_left', 0)
    return lat, lon