# main_3D_processor_V7.py
# Upgraded version with direct CV5-optimized export AND Phase 5 Data Preparation
# Updated with SVD Filter module (Phase 2)

import numpy as np
import os
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import shutil
import glob
import re
from skimage.transform import resize

# --- Using all existing, proven helper modules ---
from Ext_Data import get_external_data_paths, load_config, save_config
from data_loader import load_mlc_data, parse_radar_parameters, get_pixel_geo_coord
from phase_1_preprocessing import transform_to_frequency_domain
from phase_1_autofocus import apply_autofocus
from phase_2_subaperture import generate_sub_aperture_slcs
from phase_2_micromotion2 import estimate_micro_motions_sliding_master
from phase_2_filtering import apply_kalman_filter
# --- NEW: Import SVD Filter ---
from phase_2_svd import apply_svd_filter 
# ------------------------------
from phase_3_base_tomography import focus_sonic_tomogram 
from phase_3_modal_analysis import perform_modal_analysis
from phase_3_coherence import calculate_coherence_map
from seismic_utils import estimate_material_velocity

# --- NEW: Import the image handling module ---
from phase_image_handling import run_image_selection_pipeline
# --- NEW: Import Phase 5 Library (Ready for Future Use) ---
import phase_5_advanced_filters as p5

# --- HELPER FUNCTIONS ---

def plot_tomographic_line(ax, line_data, col_idx, start_pixel, center_row):
    ax.clear()
    magnitude = np.abs(line_data)
    pixel_rows = np.arange(start_pixel, start_pixel + len(line_data))
    ax.plot(pixel_rows, magnitude)
    ax.axvline(x=center_row, color='r', linestyle='--', label=f'POI Row ({center_row})')
    ax.set_title(f'Magnitude of Tomographic Line for Column {col_idx}')
    ax.set_xlabel('Pixel Row Index')
    ax.set_ylabel('Signal Magnitude')
    ax.grid(True)
    ax.legend()
    ax.figure.canvas.draw()

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def get_interactive_parameters(defaults, title, param_keys, full_config):
    print(f"\n--- {title} ---", flush=True)
    
    velocity_table = [
        "| Material        | Velocity (m/s) |",
        "|-----------------|----------------|",
        "| Air             | 330            |",
        "| Water           | 1500           |",
        "| Ice             | 1600 - 3500    |",
        "| Soil (Dry)      | 300 - 1000     |",
        "| Clay            | 1000 - 2500    |",
        "| Sandstone       | 2000 - 4500    |",
        "| Limestone       | 3500 - 6000    |",
        "| Concrete        | 3500 - 4500    |",
        "| Granite         | 5500 - 6000    |",
        "| Steel           | 5900           |"
    ]
    
    explanations = {
        "TOMOGRAPHY_MODE": {"options": "['FixedVelocity, (1)', 'VelocitySpectrum, (2)', 'LayeredInversion, (3)']", "desc": "Mode of tomographic inversion. Input 1, 2, or 3.", "effect": "1: FixedVelocity, 2: VelocitySpectrum, 3: LayeredInversion."},
        "SEISMIC_VELOCITY_MS": {"desc": "Fixed Speed of seismic waves (m/s), used only in 'FixedVelocity' mode. Crucial for accurate depth focusing.", "table": velocity_table, "effect": "Effect: An incorrect velocity will cause the reconstructed image to be blurred or shifted in depth."},
        "V_MIN_MS": {"desc": "Minimum velocity for the Velocity Spectrum search (m/s). Applies to: VelocitySpectrum, LayeredInversion.", "effect": "Effect: Sets the lower bound of the velocity range tested."},
        "V_MAX_MS": {"desc": "Maximum velocity for the Velocity Spectrum search (m/s). Applies to: VelocitySpectrum, LayeredInversion.", "effect": "Effect: Sets the upper bound of the velocity range tested."},
        "V_STEPS": {"desc": "Number of discrete steps to test between V_MIN_MS and V_MAX_MS. Applies to: VelocitySpectrum, LayeredInversion.", "effect": "Effect: Higher steps increase accuracy but significantly increase processing time."},
        "INVESTIGATION_FREQUENCY_HZ": {"desc": "Dominant vibration frequency (Hz). Enter value or 'auto' for detection.", "effect": "Applies to: All modes. A mismatch results in low SNR."},
        "NUM_LOOKS": {"desc": "Number of Doppler sub-apertures for vibration analysis.", "effect": "Applies to: All modes (Phase 2). More looks improve SNR."},
        "OVERLAP_FACTOR": {"desc": "Overlap factor between sub-apertures (0.0 to <1.0).", "effect": "Applies to: All modes (Phase 2). Affects number of independent samples."},
        "KALMAN_PROCESS_NOISE": {"desc": "Process noise (Q) for Kalman filter.", "effect": "Applies to: Phase 2 Filtering. Controls adaptability to signal changes."},
        "KALMAN_MEASUREMENT_NOISE": {"desc": "Measurement noise (R) for Kalman filter.", "effect": "Applies to: Phase 2 Filtering. Controls smoothing aggression."},
        "MODAL_POWER_THRESHOLD": {"desc": "Minimum normalized power for a valid frequency peak in modal analysis (0.0-1.0).", "effect": "Applies to: Modal Analysis (Phase 3). Threshold for recognizing modes."},
        "TOMO_Z_MIN_M": {"desc": "Minimum (deepest) altitude for the tomogram in meters, relative to the ground.", "effect": "Applies to: All modes. Sets the lower bound of the volume."},
        "TOMO_Z_MAX_M": {"desc": "Maximum (highest) altitude for the tomogram in meters, relative to the ground.", "effect": "Applies to: All modes. Sets the upper bound of the volume."},
        "CS_NOISE_EPSILON": {"desc": "Noise tolerance for Compressed Sensing (CS) reconstruction.", "effect": "Applies to: FixedVelocity (if CS is chosen), LayeredInversion. Controls sparsity vs data fidelity."},
        "APPLY_LOG_SCALING": {"options": "[True, False]", "desc": "Apply logarithmic scaling to the initial 1D line.", "effect": "Applies to: All modes (Phase 1). Enhances weaker signals."},
        "APPLY_WINDOWING": {"options": "[True, False]", "desc": "Apply a Hanning window to the vibration data before focusing.", "effect": "Applies to: All modes (Phase 3). Reduces sidelobe artifacts."},
        "APPLY_AUTOFOCUS": {"options": "[True, False]", "desc": "Apply a SAR autofocus algorithm to correct phase errors across the image.", "effect": "Applies to: All modes (Pre-Phase 1). Corrects platform/atmospheric phase errors."},
        "AUTOFOCUS_ITERATIONS": {"desc": "Number of iterations for the autofocus algorithm.", "effect": "Applies to: Pre-Phase 1. More iterations improve correction but increase time."},
        "APPLY_SPATIAL_FILTER": {"options": "[True, False]", "desc": "Apply a simple moving average filter to the initial 1D data line to reduce noise.", "effect": "Applies to: All modes (Phase 1). Reduces high-frequency noise."},
        "SPATIAL_FILTER_SIZE": {"desc": "The kernel size (in pixels) for the moving average spatial filter.", "effect": "Applies to: Phase 1. Larger size provides more smoothing."},
        "APPLY_KALMAN_FILTER": {"options": "[True, False]", "desc": "Apply a Kalman filter to smooth the raw vibration data over the sub-apertures.", "effect": "Applies to: All modes (Phase 2). Smoothes raw motion data."},
        "APPLY_SVD_FILTER": {"options": "[True, False]", "desc": "Apply SVD filtering to remove dominant vertical stripes (static clutter).", "effect": "Applies to: Phase 2. Removes artifacts before focusing."},
        "SVD_NUM_COMPONENTS": {"desc": "Number of dominant SVD components to remove (usually 1 for simple stripes).", "effect": "Applies to: Phase 2 SVD. Higher values remove more signal."},
        "RUN_ALL_FOCUSING_METHODS": {"options": "[True, False]", "desc": "If True, runs all methods (Beamforming, Capon, CS) for comparison.", "effect": "Applies to: FixedVelocity only."},
        "FINAL_METHOD": {"options": "['Beamforming', 'Capon', 'CS']", "desc": "The primary reconstruction method for the final output tomogram.", "effect": "Applies to: FixedVelocity only."},
        "COMPUTE_BASELINE_TOMOGRAM": {"options": "[True, False]", "desc": "A boolean flag to enable or disable the computation of a baseline tomogram for comparison.", "effect": "Applies to: All modes (Output)."},
        "SEISMIC_DAMPING_COEFF": {"desc": "Seismic damping coefficient used in the focusing model.", "effect": "Applies to: All Tomography Modes (Phase 3). Accounts for signal attenuation."},
        "PERFORM_MODAL_ANALYSIS": {"options": "[True, False]", "desc": "Perform modal analysis to map dominant vibration frequencies across the analysis line.", "effect": "Applies to: All modes (Phase 3)."}
    }
    
    TOMO_MODE_MAP = {1: "FixedVelocity", 2: "VelocitySpectrum", 3: "LayeredInversion"}
    
    def get_typed_value(key, str_value, default_value):
        if isinstance(default_value, bool):
            return str_value.lower() in ['true', 't', '1', 'yes', 'y']
        if isinstance(default_value, int):
            try: return int(str_value)
            except ValueError: return default_value
        if isinstance(default_value, float):
            try: return float(str_value)
            except ValueError: return default_value
        return str_value

    params = defaults.copy()
    if 'user_parameters' in full_config:
        saved_params = full_config['user_parameters']
        
        saved_params_ci = {}
        for key, value in saved_params.items():
            saved_params_ci[key] = value
            saved_params_ci[key.upper()] = value
            saved_params_ci[key.lower()] = value
        
        for key in param_keys:
            if key in saved_params_ci:
                saved_value = saved_params_ci[key]
                if key in params:
                    params[key] = get_typed_value(key, saved_value, params[key])
            elif key.upper() in saved_params_ci:
                saved_value = saved_params_ci[key.upper()]
                if key in params:
                    params[key] = get_typed_value(key, saved_value, params[key])
            elif key.lower() in saved_params_ci:
                saved_value = saved_params_ci[key.lower()]
                if key in params:
                    params[key] = get_typed_value(key, saved_value, params[key])
    
    print("\nCurrent Configuration:")
    print("-" * 50)
    for key in param_keys:
        if key in params:
            val = params[key]
            if key == "TOMOGRAPHY_MODE":
                val = f"{val} ({[k for k,v in TOMO_MODE_MAP.items() if v==val][0]})"
            print(f"  {key:<30}: {val}")
    print("-" * 50)

    print("\nNOTE: You can accept defaults by pressing Enter.")
    if input("Do you want to change any of these parameters? (y/N): ").lower() in ['y', 'yes']:
        for key in param_keys:
            if key not in params: continue
            original_value = params[key]
            explanation = explanations.get(key, {})
            print("-" * 20, flush=True)
            if key == "TOMOGRAPHY_MODE":
                print(f"Parameter: {key}\nDescription: {explanation['desc']}\nValid options: {explanation['options']}")
                current_mode_num = [k for k, v in TOMO_MODE_MAP.items() if v == original_value][0]
                new_val_str = input(f"Enter option (1-3) [Default: {original_value} ({current_mode_num})]: ")
                if new_val_str:
                    try:
                        mode_num = int(new_val_str)
                        if mode_num in TOMO_MODE_MAP:
                            params[key] = TOMO_MODE_MAP[mode_num]
                    except: pass
                continue
            
            print(f"Parameter: {key}")
            if "desc" in explanation: print(f"Description: {explanation['desc']}")
            if "table" in explanation:
                print("\nReference Velocities:")
                for line in explanation['table']: print(line)
            if "effect" in explanation: print(f"{explanation['effect']}")
            
            new_val_str = input(f"Enter new value [Default: {original_value}]: ")
            if new_val_str:
                if key == "INVESTIGATION_FREQUENCY_HZ" and new_val_str.lower() == 'auto':
                    params[key] = 'auto'; continue
                try:
                    new_value = get_typed_value(key, new_val_str, original_value)
                    if new_value != original_value:
                        params[key] = new_value
                except Exception:
                    pass

    print(f"\nSaving all current parameters to configuration for next run...")
    if 'user_parameters' not in full_config:
        full_config['user_parameters'] = {}
    
    for key, value in params.items():
        full_config['user_parameters'][key] = str(value)
    
    try:
        save_config(full_config)
    except Exception as e:
        print(f"\nWarning: Failed to save configuration: {e}")

    return params


# --- MAIN EXECUTION FUNCTION ---
def main():
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"tomography_3D_results_{run_timestamp}.npz"
    temp_dir = f"temp_results_{run_timestamp}"
    
    print(f"\n--- 3D Seismic Tomography Processor V7 ---")
    print(f"Output for this run will be saved to: {output_filename}", flush=True)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    try:
        full_config, _ = load_config()
        if not full_config:
            full_config = {}

        file_paths_base = get_external_data_paths()
        if not file_paths_base:
            return
        
        base_name = os.path.splitext(file_paths_base['tiff_file'])[0]
        txt_files = glob.glob(f'{base_name}_metadata.txt')
        file_paths = file_paths_base
        if txt_files:
            file_paths['txt_file'] = txt_files[0]
        else: 
            print(f"Warning: No matching *_metadata.txt found for {base_name}.")
            file_paths['txt_file'] = None

        # --- RUN IMAGE SELECTION PIPELINE ---
        print("\n--- Starting Image Selection Pipeline ---")
        complex_data_full, radar_params, user_selection, center_row, central_col_idx = run_image_selection_pipeline(
            file_paths, full_config, run_timestamp
        )
        
        if complex_data_full is None:
            print("Image selection pipeline failed or was cancelled.")
            return

        # --- END OF SELECTION LOOP ---

        default_params = {
            "TOMOGRAPHY_MODE": "FixedVelocity", 
            "SEISMIC_VELOCITY_MS": 4000.0, 
            "V_MIN_MS": 2000.0, 
            "V_MAX_MS": 6000.0, 
            "V_STEPS": 20, 
            "INVESTIGATION_FREQUENCY_HZ": 20.0,
            "MICROMOTION_METHOD": "SlidingMaster", 
            "APPLY_LOG_SCALING": True, 
            "APPLY_WINDOWING": True,
            "APPLY_AUTOFOCUS": False, 
            "AUTOFOCUS_ITERATIONS": 5, 
            "APPLY_SPATIAL_FILTER": True,
            "SPATIAL_FILTER_SIZE": 5, 
            "APPLY_KALMAN_FILTER": False, 
            "KALMAN_PROCESS_NOISE": 0.2, 
            "KALMAN_MEASUREMENT_NOISE": 0.1, 
            "APPLY_SVD_FILTER": False, 
            "SVD_NUM_COMPONENTS": 1,
            "FINAL_METHOD": "Beamforming", 
            "RUN_ALL_FOCUSING_METHODS": False,
            "PERFORM_MODAL_ANALYSIS": True, 
            "MODAL_POWER_THRESHOLD": 0.1, 
            "TOMO_Z_MIN_M": -2000, 
            "TOMO_Z_MAX_M": 500, 
            "CS_NOISE_EPSILON": 0.15, 
            "NUM_LOOKS": 96, 
            "OVERLAP_FACTOR": 0.85, 
            "COMPUTE_BASELINE_TOMOGRAM": True, 
            "SEISMIC_DAMPING_COEFF": 0.05
        }
        
        param_order = [
            "TOMOGRAPHY_MODE", "SEISMIC_VELOCITY_MS", "V_MIN_MS", "V_MAX_MS", "V_STEPS",
            "INVESTIGATION_FREQUENCY_HZ", "NUM_LOOKS", "OVERLAP_FACTOR", 
            "FINAL_METHOD", "RUN_ALL_FOCUSING_METHODS", "TOMO_Z_MIN_M", "TOMO_Z_MAX_M", 
            "CS_NOISE_EPSILON", "SEISMIC_DAMPING_COEFF", "APPLY_LOG_SCALING", "APPLY_WINDOWING", 
            "PERFORM_MODAL_ANALYSIS", "MODAL_POWER_THRESHOLD", "APPLY_SPATIAL_FILTER", 
            "SPATIAL_FILTER_SIZE", "APPLY_KALMAN_FILTER", "KALMAN_PROCESS_NOISE", 
            "KALMAN_MEASUREMENT_NOISE", "APPLY_SVD_FILTER", "SVD_NUM_COMPONENTS", 
            "APPLY_AUTOFOCUS", "AUTOFOCUS_ITERATIONS", 
            "COMPUTE_BASELINE_TOMOGRAM"
        ]

        params = get_interactive_parameters(default_params, "Seismic Tomography Parameters", param_order, full_config)
        
        params['ANALYSIS_EXTENT_KM'] = user_selection['analysis_extent_km']

        print(f"\n--- Saving all user selections for next run ---")
        full_config['user_parameters']['ANALYSIS_EXTENT_KM'] = str(user_selection['analysis_extent_km'])
        full_config['column_ranges'] = {
            'start_left': str(user_selection['start_left']),
            'end_left': str(user_selection['end_left']),
            'start_right': str(user_selection['start_right']),
            'end_right': str(user_selection['end_right'])
        }
        
        for key, value in params.items():
            full_config['user_parameters'][key] = str(value)
        
        try:
            save_config(full_config)
            print("✓ All selections saved as defaults for next run.")
        except Exception as e:
            print(f"Warning: Failed to save configuration: {e}")
        
        complex_data = complex_data_full

        center_set = {central_col_idx}
        
        l_start_offset = user_selection['start_left']
        l_end_offset = user_selection['end_left']
        left_min = central_col_idx - l_end_offset
        left_max = central_col_idx - l_start_offset
        left_range = list(range(left_min, left_max + 1))
        
        r_start_offset = user_selection['start_right']
        r_end_offset = user_selection['end_right']
        right_min = central_col_idx + r_start_offset
        right_max = central_col_idx + r_end_offset
        right_range = list(range(right_min, right_max + 1))
        
        cols_to_process = sorted(list(set(left_range) | center_set | set(right_range)))
        cols_to_process = [c for c in cols_to_process if 0 <= c < complex_data.shape[1]]
        
        if not cols_to_process: 
            print("No valid columns selected.")
            return
        
        print("\n--- Processing Plan ---")
        print(f"  Center Column: {central_col_idx}")
        print(f"  Left Strip ({l_start_offset}-{l_end_offset}):  {left_min} to {left_max} (Count: {len(left_range)})")
        print(f"  Right Strip ({r_start_offset}-{r_end_offset}): {right_min} to {right_max} (Count: {len(right_range)})")
        print(f"  Total Unique Columns to Process: {len(cols_to_process)}")
        
        spacing_m = radar_params.get('azimuth_spacing_m', 1.0)
        extent_km = user_selection["analysis_extent_km"]
        extent_in_pixels = int(round((extent_km * 1000) / spacing_m))
        start_pixel = max(0, center_row - extent_in_pixels)
        end_pixel = min(radar_params['scene_rows'] - 1, center_row + extent_in_pixels)
        
        if params.get("APPLY_AUTOFOCUS", False):
            complex_data = apply_autofocus(complex_data, params["AUTOFOCUS_ITERATIONS"])

        z_vec, processed_count = None, 0
        print(f"\nProcessing {len(cols_to_process)} columns from {min(cols_to_process)} to {max(cols_to_process)}.")
        
        vibration_frequency_hz = params["INVESTIGATION_FREQUENCY_HZ"]
        if isinstance(vibration_frequency_hz, str) and vibration_frequency_hz.lower() == 'auto':
            print("Auto-detecting Investigation Frequency... Using fallback value of 20.0 Hz.")
            vibration_frequency_hz = 20.0
        
        interrupted = False

        try:
            for i, current_col_idx in enumerate(cols_to_process):
                print(f"\n--- Processing Column {current_col_idx} ({i+1}/{len(cols_to_process)}) ---", flush=True)
                if start_pixel >= complex_data.shape[0] or current_col_idx >= complex_data.shape[1]: 
                    continue
                
                tomographic_line_raw = complex_data[start_pixel:end_pixel+1, current_col_idx]
                
                if np.max(np.abs(tomographic_line_raw)) < 1e-9: 
                    print("No signal, skipping")
                    continue
                
                tomographic_line_main = np.copy(tomographic_line_raw)
                
                if params.get("APPLY_SPATIAL_FILTER", False):
                    filter_size = params.get("SPATIAL_FILTER_SIZE", 5)
                    tomographic_line_main = np.convolve(tomographic_line_main, np.ones(filter_size)/filter_size, 'same')
                
                if params.get("APPLY_LOG_SCALING", False):
                    tomographic_line_main = np.log1p(np.abs(tomographic_line_main)) * np.exp(1j * np.angle(tomographic_line_main))

                sub_ap_args = {
                    "num_looks": params["NUM_LOOKS"], 
                    "overlap_factor": params["OVERLAP_FACTOR"]
                }
                
                if radar_params.get('doppler_params_available', False):
                    sub_ap_args.update({
                        "doppler_bandwidth_hz": radar_params.get('doppler_bandwidth_hz'),
                        "doppler_centroid_hz": radar_params.get('doppler_centroid_initial_hz', 0.0),
                        "doppler_ambiguity_spacing_hz": radar_params.get('doppler_ambiguity_spacing_hz'),
                        "max_unambiguous_doppler_hz": radar_params.get('max_unambiguous_doppler_hz'),
                        "sar_metadata": radar_params
                    })
                    print(f"[DOPPLER] Passing Doppler parameters and metadata to sub-aperture generation")
                else:
                    print(f"[DOPPLER] Using default sub-aperture generation (no Doppler parameters)")
                
                low_res_slcs, sub_ap_centers = generate_sub_aperture_slcs(
                    transform_to_frequency_domain(tomographic_line_main), 
                    **sub_ap_args
                )
                
                if len(low_res_slcs) < 2: 
                    print("Not enough sub-apertures, skipping")
                    continue

                Y_processed = estimate_micro_motions_sliding_master(low_res_slcs)
                if Y_processed.size == 0: 
                    print("No micro-motions estimated, skipping")
                    continue
                
                num_looks_for_focusing = Y_processed.shape[1]
                sub_ap_centers = sub_ap_centers[:num_looks_for_focusing]
                
                if params.get("APPLY_KALMAN_FILTER", False): 
                    Y_processed = apply_kalman_filter(Y_processed, params["KALMAN_PROCESS_NOISE"], params["KALMAN_MEASUREMENT_NOISE"])

                # --- NEW SVD FILTER ---
                if params.get("APPLY_SVD_FILTER", False):
                    Y_processed = apply_svd_filter(Y_processed, n_components=params.get("SVD_NUM_COMPONENTS", 1))
                # ----------------------

                tomogram_stages = {}
                focus_args = {
                    "num_looks": num_looks_for_focusing, 
                    "sub_ap_centers": sub_ap_centers, 
                    "seismic_velocity_ms": params["SEISMIC_VELOCITY_MS"], 
                    "vibration_frequency_hz": vibration_frequency_hz, 
                    "apply_windowing": params.get("APPLY_WINDOWING", True), 
                    "z_min": params["TOMO_Z_MIN_M"], 
                    "z_max": params["TOMO_Z_MAX_M"], 
                    "epsilon": params["CS_NOISE_EPSILON"], 
                    "damping_coeff": params["SEISMIC_DAMPING_COEFF"],
                    "method": params["TOMOGRAPHY_MODE"], 
                    "final_method": params["FINAL_METHOD"],
                    "v_min": params["V_MIN_MS"], 
                    "v_max": params["V_MAX_MS"], 
                    "v_steps": params["V_STEPS"]
                }
                
                try:
                    # NEW: Receive Complex Data
                    tomo_complex, z_vec_current, third_output_array = focus_sonic_tomogram(Y_processed, radar_params, tomographic_line_main.shape, **focus_args)
                    
                    # Convert to magnitude for standard display/saving
                    tomo_final = np.abs(tomo_complex)
                    
                    velocity_map_2d = third_output_array if params["TOMOGRAPHY_MODE"] == 'VelocitySpectrum' else np.zeros((tomo_final.shape[0], tomo_final.shape[1]), dtype=np.float32)
                    robustness_score_2d = third_output_array if params["TOMOGRAPHY_MODE"] == 'LayeredInversion' else np.zeros((tomo_final.shape[0], tomo_final.shape[1]), dtype=np.float32)

                    if params["TOMOGRAPHY_MODE"] == 'LayeredInversion':
                        r_scores_valid = robustness_score_2d[robustness_score_2d > 0]
                        avg_r_score = np.mean(r_scores_valid) if r_scores_valid.size > 0 else 0.0
                        print(f"    Average Model Robustness Index (R-Score): {avg_r_score:.2f} / 10.0")
                    
                    if params["TOMOGRAPHY_MODE"] == 'FixedVelocity':
                        methods_to_run = ['beamforming', 'capon', 'cs'] if params.get("RUN_ALL_FOCUSING_METHODS", False) else [params["FINAL_METHOD"].lower()]
                        for method_name in methods_to_run:
                            tomo_stage_complex, _, _ = focus_sonic_tomogram(Y_processed, radar_params, tomographic_line_main.shape, method='FixedVelocity', final_method=method_name.capitalize(), **{k:v for k,v in focus_args.items() if k not in ['method', 'final_method']})
                            tomogram_stages[method_name.capitalize()] = np.abs(tomo_stage_complex)
                        tomo_final = tomogram_stages[params["FINAL_METHOD"].capitalize()]
                    else:
                        tomogram_stages[params["TOMOGRAPHY_MODE"]] = tomo_final
                        
                    if z_vec is None: z_vec = z_vec_current
                    
                    coherence_map = calculate_coherence_map(Y_processed)
                    freq_map, power_map, mode_map = None, None, None
                    if params.get("PERFORM_MODAL_ANALYSIS", False): 
                        freq_map, power_map, mode_map = perform_modal_analysis(Y_processed, params["SEISMIC_VELOCITY_MS"], 146.5, num_looks=num_looks_for_focusing, power_threshold=params["MODAL_POWER_THRESHOLD"])

                    slice_filename = os.path.join(temp_dir, f"slice_{current_col_idx}.npz")
                    np.savez_compressed(slice_filename, 
                                        tomogram_final=tomo_final, 
                                        tomogram_complex=tomo_complex, # SAVE COMPLEX DATA
                                        tomogram_stages=tomogram_stages, 
                                        coherence_map=coherence_map, 
                                        frequency_map=freq_map, 
                                        power_map=power_map, 
                                        mode_number_map=mode_map, 
                                        z_vec=z_vec, 
                                        low_res_slcs=low_res_slcs, 
                                        y_matrix_processed=Y_processed, 
                                        velocity_map_2d=velocity_map_2d, 
                                        robustness_score_2d=robustness_score_2d)
                    processed_count += 1
                
                except Exception as col_error:
                    print(f"❌ Error processing column {current_col_idx}: {col_error}")
                    continue

        except KeyboardInterrupt:
            print("\n\n--- KEYBOARD INTERRUPT DETECTED ---", flush=True)
            print("Stopping processing loop and proceeding to save partial results...", flush=True)
            interrupted = True
        
        if processed_count == 0: 
            if interrupted:
                print("No columns were fully processed before interrupt. No output file will be generated.", flush=True)
            else:
                print("No columns were successfully processed.", flush=True)
            return
        
        print(f"\n--- Assembling final 3D data cube... ---")
        if interrupted:
            print("--- NOTE: Assembling PARTIAL results due to user interrupt. ---", flush=True)
            
        tomogram_list, tomogram_complex_list = [], []
        stages_list, coherence_list, freq_list, power_list, mode_list, low_res_slcs_list, y_matrix_list, velocity_map_list, robustness_list = [],[],[],[],[],[],[],[],[]
        processed_cols_final = []
        sorted_files = sorted(glob.glob(os.path.join(temp_dir, "slice_*.npz")), key=lambda f: int(re.search(r'slice_(\d+).npz', f).group(1)))
        
        for f in sorted_files:
            col_num = int(re.search(r'slice_(\d+).npz', f).group(1)); processed_cols_final.append(col_num)
            with np.load(f, allow_pickle=True) as d:
                tomogram_list.append(d['tomogram_final'])
                if 'tomogram_complex' in d: tomogram_complex_list.append(d['tomogram_complex'])
                stages_list.append(d['tomogram_stages'].item())
                coherence_list.append(d['coherence_map'])
                if 'frequency_map' in d and d['frequency_map'] is not None: freq_list.append(d['frequency_map'])
                if 'power_map' in d and d['power_map'] is not None: power_list.append(d['power_map'])
                if 'mode_number_map' in d and d['mode_number_map'] is not None: mode_list.append(d['mode_number_map'])
                if 'robustness_score_2d' in d and d['robustness_score_2d'] is not None: robustness_list.append(d['robustness_score_2d'])
                if z_vec is None: z_vec = d['z_vec']
                if 'low_res_slcs' in d: low_res_slcs_list.append(d['low_res_slcs'])
                if 'y_matrix_processed' in d: y_matrix_list.append(d['y_matrix_processed'])
                if 'velocity_map_2d' in d:
                    velocity_map_list.append(d['velocity_map_2d'])
                else:
                    velocity_map_list.append(np.zeros_like(d['tomogram_final'], dtype=np.float32))

        print(f"\n--- Finalizing Geographic Parameters ---")
        required_geo_keys = ['lat_upper_left', 'lon_upper_left', 'lat_lower_left', 'lon_lower_left', 
                           'lat_upper_right', 'lon_upper_right', 'lat_lower_right', 'lon_lower_right',
                           'scene_rows', 'scene_cols']
        
        missing_geo_keys = []
        for key in required_geo_keys:
            if key not in radar_params:
                missing_geo_keys.append(key)
                if key == 'scene_rows': radar_params[key] = complex_data.shape[0]
                elif key == 'scene_cols': radar_params[key] = complex_data.shape[1]
                else: radar_params[key] = 0.0 # Fallback
        
        if missing_geo_keys:
            print(f"⚠️  GEOGRAPHIC WARNING: Missing keys {missing_geo_keys}. Creating synthetic grid.")
            base_lat = radar_params.get('lat_start', 45.0)
            base_lon = radar_params.get('lon_start', -93.0)
            lat_delta_per_pixel = 1.0 / 111000  # ~1 meter in degrees
            lon_delta_per_pixel = 1.0 / (111000 * np.cos(np.radians(base_lat)))
            
            radar_params.update({
                'lat_upper_left': base_lat,
                'lon_upper_left': base_lon,
                'lat_lower_left': base_lat - (radar_params['scene_rows'] * lat_delta_per_pixel),
                'lon_lower_left': base_lon,
                'lat_upper_right': base_lat,
                'lon_upper_right': base_lon + (radar_params['scene_cols'] * lon_delta_per_pixel),
                'lat_lower_right': base_lat - (radar_params['scene_rows'] * lat_delta_per_pixel),
                'lon_lower_right': base_lon + (radar_params['scene_cols'] * lon_delta_per_pixel),
            })

        # =====================================================================
        # CRITICAL FIXES FOR VISUALIZER COMPATIBILITY
        # =====================================================================
        
        if tomogram_list:
            target_shape = tomogram_list[0].shape
            print(f"Standardizing all slices to shape: {target_shape}")
            
            def standardize_shape(arr, target_shape):
                arr_dim = arr.ndim
                target_dim = len(target_shape)
                if arr_dim == 1 and target_dim == 2:
                    expanded = np.zeros(target_shape, dtype=arr.dtype)
                    for col in range(target_shape[1]):
                        expanded[:, col] = arr[:]
                    return expanded
                elif arr_dim == 2 and target_dim == 2:
                    if arr.shape == target_shape:
                        return arr
                    result = np.zeros(target_shape, dtype=arr.dtype)
                    min_rows = min(arr.shape[0], target_shape[0])
                    min_cols = min(arr.shape[1], target_shape[1])
                    result[:min_rows, :min_cols] = arr[:min_rows, :min_cols]
                    return result
                else:
                    return arr
            
            tomogram_list = [standardize_shape(t, target_shape) for t in tomogram_list]
            if tomogram_complex_list:
                tomogram_complex_list = [standardize_shape(t, target_shape) for t in tomogram_complex_list]
            coherence_list = [standardize_shape(c, target_shape) for c in coherence_list]
            if velocity_map_list:
                velocity_map_list = [standardize_shape(v, target_shape) for v in velocity_map_list]
            if robustness_list:
                robustness_list = [standardize_shape(r, target_shape) for r in robustness_list]
        
        # Build cubes with (pixels, cols, depth) orientation
        tomogram_cube_raw = np.stack(tomogram_list, axis=0)  # (cols, pixels, depth)
        tomogram_cube_3d = np.transpose(tomogram_cube_raw, (1, 0, 2))  # (pixels, cols, depth)
        
        # NEW: Build Complex Cube
        if tomogram_complex_list:
            tomogram_complex_raw = np.stack(tomogram_complex_list, axis=0)
            tomogram_cube_complex_3d = np.transpose(tomogram_complex_raw, (1, 0, 2))
        else:
            tomogram_cube_complex_3d = np.zeros_like(tomogram_cube_3d, dtype=np.complex64)

        coherence_cube_raw = np.stack(coherence_list, axis=0)
        coherence_cube_3d = np.transpose(coherence_cube_raw, (1, 0, 2))
        
        if velocity_map_list:
            velocity_cube_raw = np.stack(velocity_map_list, axis=0)
            velocity_cube_3d = np.transpose(velocity_cube_raw, (1, 0, 2))
        else:
            velocity_cube_3d = np.zeros_like(tomogram_cube_3d, dtype=np.float32)
        
        if robustness_list:
            robustness_cube_raw = np.stack(robustness_list, axis=0)
            robustness_cube_3d = np.transpose(robustness_cube_raw, (1, 0, 2))
        else:
            robustness_cube_3d = np.zeros_like(tomogram_cube_3d, dtype=np.float32)
        
        frequency_map_2d = np.stack(freq_list, axis=0) if freq_list else np.zeros((len(processed_cols_final), tomogram_list[0].shape[0]), dtype=np.float32)
        power_map_2d = np.stack(power_list, axis=0) if power_list else np.zeros((len(processed_cols_final), tomogram_list[0].shape[0]), dtype=np.float32)
        mode_number_map_2d = np.stack(mode_list, axis=0) if mode_list else np.zeros((len(processed_cols_final), tomogram_list[0].shape[0]), dtype=np.float32)
        
        print("\n--- Calculating geographic coordinates for visualization ---")
        analysis_extent_km = user_selection['analysis_extent_km']
        range_spacing_m = radar_params.get('range_spacing_m', 1.0)
        extent_in_pixels_horizontal = int(round((analysis_extent_km * 1000) / range_spacing_m))
        left_column = max(0, central_col_idx - extent_in_pixels_horizontal)
        right_column = min(radar_params['scene_cols'] - 1, central_col_idx + extent_in_pixels_horizontal)

        geo_start_top = get_pixel_geo_coord(start_pixel, left_column, radar_params)
        geo_end_bottom = get_pixel_geo_coord(end_pixel, right_column, radar_params)
        geo_left_center = get_pixel_geo_coord(center_row, left_column, radar_params)
        geo_right_center = get_pixel_geo_coord(center_row, right_column, radar_params)

        analysis_line_length_m = haversine_distance(
            geo_left_center[0], geo_left_center[1], 
            geo_right_center[0], geo_right_center[1]
        )

        print(f"  Analysis volume diagonal: ({start_pixel},{left_column}) to ({end_pixel},{right_column})")
        print(f"  Top-left (geo_start): {geo_start_top}")
        print(f"  Bottom-right (geo_end): {geo_end_bottom}")
        print(f"  Horizontal distance: {analysis_line_length_m:.1f} meters")
        
        processed_cols_array = np.array(processed_cols_final)
        
        # --- NEW: CALCULATE VOXEL SPACING ---
        # [dz, dy (azimuth), dx (range)]
        dz_meters = abs(z_vec[1] - z_vec[0]) if len(z_vec) > 1 else 1.0
        dy_meters = radar_params.get('azimuth_spacing_m', 1.0)
        dx_meters = radar_params.get('range_spacing_m', 1.0)
        voxel_spacing_meters = np.array([dz_meters, dy_meters, dx_meters])
        print(f"  Voxel Spacing (Z, Y, X): {voxel_spacing_meters}")

        # FIX 2: Store all relevant coordinates
        results_to_save = {
            'source_data_file': os.path.basename(file_paths['tiff_file']), 
            'central_column_index': central_col_idx, 
            'processed_column_indices': processed_cols_array,
            'vertical_pixel_range': np.array([start_pixel, end_pixel]), 
            'radar_params': radar_params, 
            'user_parameters': params, 
            'z_vec': z_vec,
            'tomography_mode': params.get("TOMOGRAPHY_MODE", "FixedVelocity"),
            'geo_start': geo_start_top,
            'geo_end': geo_end_bottom,
            'geo_left_center': geo_left_center,
            'geo_right_center': geo_right_center,
            'analysis_line_length_m': analysis_line_length_m,
            'vertical_extent_m': abs(end_pixel - start_pixel) * radar_params.get('azimuth_spacing_m', 1.0),
            'horizontal_extent_m': abs(right_column - left_column) * radar_params.get('range_spacing_m', 1.0),
            'tomogram_cube': tomogram_cube_3d,
            'tomogram_cube_complex': tomogram_cube_complex_3d, # SAVED HERE
            'coherence_cube': coherence_cube_3d,
            'velocity_map_3d': velocity_cube_3d,
            'robustness_score_3d': robustness_cube_3d,
            'frequency_map_2d': frequency_map_2d,
            'power_map_2d': power_map_2d,
            'mode_number_map_2d': mode_number_map_2d,
            'tomogram_stages_list': np.array(stages_list, dtype=object),
            'voxel_spacing_meters': voxel_spacing_meters # SAVED HERE
        }
        
        if low_res_slcs_list: 
            results_to_save['low_res_slcs_list'] = np.array(low_res_slcs_list, dtype=object)
        if y_matrix_list: 
            results_to_save['y_matrix_processed_3d'] = np.array(y_matrix_list, dtype=object)
        
        print("\n" + "="*70)
        print("V7 ENHANCEMENT: ADDING CV5-COMPATIBLE FIELD NAMES")
        print("="*70)
        
        results_to_save['velocity_cube'] = results_to_save['velocity_map_3d']
        
        if 'x_ideal_1d' not in results_to_save:
            num_cols = results_to_save['tomogram_cube'].shape[1]
            results_to_save['x_ideal_1d'] = np.linspace(0, num_cols * 10, num_cols)
        
        if 'y_ideal_1d' not in results_to_save:
            num_pixels = results_to_save['tomogram_cube'].shape[0]
            results_to_save['y_ideal_1d'] = np.linspace(0, num_pixels * 10, num_pixels)
        
        start, end = results_to_save['vertical_pixel_range']
        num_pixels = results_to_save['tomogram_cube'].shape[0]
        results_to_save['pixel_rows'] = np.linspace(start, end, num_pixels, dtype=int)
        
        processed_cols = results_to_save['processed_column_indices']
        if hasattr(processed_cols, 'tolist'):
            results_to_save['all_cols'] = processed_cols.tolist()
        else:
            results_to_save['all_cols'] = list(processed_cols)
        
        results_to_save['is_pre_degraded'] = False
        results_to_save['ds_factor'] = 1
        results_to_save['source_file_type'] = 'main_3D_processor_V7'
        
        results_to_save['final_cube'] = results_to_save['tomogram_cube']
        results_to_save['final_coherence_cube'] = results_to_save['coherence_cube']
        results_to_save['final_velocity_cube'] = results_to_save['velocity_cube']
        results_to_save['z_vec_ref'] = results_to_save['z_vec']
        print(f"✓ Added CV5 compatibility aliases (final_cube, z_vec_ref, etc.)")
        
        np.savez_compressed(output_filename, **results_to_save)
        print(f"\n" + "="*70)
        print(f"✓ V7 RESULTS SAVED SUCCESSFULLY")
        print("="*70)
        print(f"File: {output_filename}")
        print(f"Size: {os.path.getsize(output_filename) / (1024*1024):.1f} MB")
        print(f"Cube dimensions: {results_to_save['tomogram_cube'].shape}")
        print(f"Columns processed: {len(results_to_save['all_cols'])}")
        print(f"Depth levels: {len(results_to_save['z_vec'])}")
        print(f"Tomography mode: {results_to_save['tomography_mode']}")
        print(f"CV5 compatibility: ✓ Field names added")
        print(f"Phase 5 Ready: ✓ Complex Cube, Voxel Spacing, and Y-Matrix saved.")
        print("="*70)
        
        if interrupted:
            print("\n" + "="*70)
            print("⚠ NOTE: This file contains PARTIAL results due to user interrupt.")
            print("="*70)

    except Exception as e:
        import traceback
        print(f"\n" + "="*70)
        print(f"ERROR DURING PROCESSING: {e}")
        print("="*70)
        traceback.print_exc()
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n--- Temporary directory '{temp_dir}' has been deleted ---", flush=True)

if __name__ == "__main__":
    main()