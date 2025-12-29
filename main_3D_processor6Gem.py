# main_3D_processor5.py
# =============================================================================
#   SAR 3D ECHOGRAPHY & TOMOGRAPHY PROCESSOR (V7-41: PROBE SPEED FIX)
#   CRITICAL FIXES:
#     1. PROBE SPEED: Forced Probe to use low precision (10x) for speed.
#     2. LOGGING: Suppressed "Phase 2" printouts during Probe to avoid confusion.
#     3. PERSISTENCE: Ensures calibration is saved/loaded correctly.
# =============================================================================

import sys
import os
import numpy as np
import datetime
import shutil
import glob
import re
import json

# --- GUI BACKEND CONFIGURATION ---
BACKEND = 'TkAgg' 
import matplotlib
if BACKEND != 'Default':
    try:
        matplotlib.use(BACKEND)
        print(f"[SYSTEM] Matplotlib backend set to: {BACKEND}")
    except:
        pass

import matplotlib.pyplot as plt
from skimage.transform import resize

# --- EXISTING HELPER MODULES ---
from Ext_Data import get_external_data_paths, load_config, save_config
from data_loader import load_mlc_data, parse_radar_parameters, get_pixel_geo_coord
from phase_image_handling import run_image_selection_pipeline
from phase_1_preprocessing import extract_tomographic_line, transform_to_frequency_domain
from phase_1_autofocus import apply_autofocus
from phase_2_subaperture import generate_sub_aperture_slcs
from phase_2_micromotion2 import estimate_micro_motions_sliding_master
from phase_2_svd import apply_svd_filter
from phase_2_filtering import apply_kalman_filter
from phase_3_base_tomography import focus_sonic_tomogram
from phase_3_modal_analysis import perform_modal_analysis, perform_frequency_sweep_snr
from phase_3_advanced_tomography import perform_velocity_autofocus
from phase_3_coherence import calculate_coherence_map
from phase_4_validation import geocode_tomographic_line_coords
from phase_5_advanced_filters import apply_hessian_frangi_filter_3d

# --- HELPER FUNCTIONS ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def find_key_recursive(data, target_key):
    """Deep search for a key in a nested dictionary/list structure."""
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() == target_key.lower():
                return v
            item = find_key_recursive(v, target_key)
            if item is not None:
                return item
    elif isinstance(data, list):
        for item in data:
            result = find_key_recursive(item, target_key)
            if result is not None:
                return result
    return None

def get_interactive_parameters(defaults, title, param_keys, full_config, lock_mode=False):
    """
    Displays parameters. If lock_mode is True, it strictly prevents changes.
    """
    print(f"\n--- {title} ---", flush=True)
    
    velocity_table = [
        "| Material        | Velocity (m/s) |", "|-----------------|----------------|",
        "| Air             | 330            |", "| Water           | 1500           |",
        "| Ice             | 1600 - 3500    |", "| Soil (Dry)      | 300 - 1000     |",
        "| Clay            | 1000 - 2500    |", "| Sandstone       | 2000 - 4500    |",
        "| Limestone       | 3500 - 6000    |", "| Concrete        | 3500 - 4500    |",
        "| Granite         | 5500 - 6000    |", "| Steel           | 5900           |"
    ]
    
    TOMO_MAP = {1: "FixedVelocity", 2: "VelocitySpectrum", 3: "LayeredInversion"}
    TARGET_MAP = {1: "building", 2: "geology", 3: "bridge"}
    METHOD_MAP = {1: "beamforming", 2: "capon", 3: "cs"}
    
    explanations = {
        "TOMOGRAPHY_MODE": {
            "desc": "Mode of tomographic inversion.", "map": TOMO_MAP,
            "options_text": "1: FixedVelocity (Standard), 2: VelocitySpectrum (Scan for Vel), 3: LayeredInversion (Advanced)"
        },
        "TARGET_TYPE": {
            "desc": "Target Material Logic (Sets Physics & Math Defaults).", "map": TARGET_MAP,
            "options_text": "1: Building (Concrete/CS), 2: Geology (Rock/Capon), 3: Bridge (Steel/Beamforming)",
            "details": [
                "\n   | Opt | Material | Vel     | Method      | Best For...          |",
                "   |-----|----------|---------|-------------|----------------------|",
                "   |  1  | Concrete | 555 m/s | CS (L1)     | Walls, Floors, Rooms |",
                "   |  2  | Rock     | 3000 m/s| Capon       | Voids, Magma, Soil   |",
                "   |  3  | Steel    | 5000 m/s| Beamforming | Cracks, Cables       |"
            ]
        },
        "FINAL_METHOD": {
            "desc": "Primary reconstruction algorithm.", "map": METHOD_MAP,
            "options_text": "1: Beamforming (Robust/Blurry), 2: Capon (Adaptive/Sharper), 3: CS (Compressed Sensing/Sharpest)"
        },
        "RUN_ALL_FOCUSING_METHODS": {
            "desc": "Run all solvers for comparison? If True, executes Beamforming, Capon, AND CS sequentially for benchmarking.",
            "options": "[True, False]"
        },
        "CS_NOISE_EPSILON": {
            "desc": "Noise tolerance for Compressed Sensing. Low (0.05) = High Fidelity (fits noise). High (0.5) = Forces geometric sparsity."
        },
        "SEISMIC_VELOCITY_MS": {
            "desc": "Fixed Speed of seismic waves (m/s). Incorrect velocity causes blurred or shifted depth.", "table": velocity_table
        },
        "V_MIN_MS": {
            "desc": "Min velocity for sweep. Start of search range (e.g. 300 m/s). Must be < V_MAX."
        }, 
        "V_MAX_MS": {
            "desc": "Max velocity for sweep. End of search range (e.g. 6000 m/s). Limits calculation time."
        },
        "V_STEPS": {"desc": "Number of velocity steps."},
        "INVESTIGATION_FREQUENCY_HZ": {
            "desc": "Dominant vibration frequency (Hz). The specific frequency to focus on. Mismatch = Signal Loss."
        },
        "NUM_LOOKS": {
            "desc": "Number of Doppler sub-apertures. More looks = Smoother time series (>500 for Echography)."
        },
        "OVERLAP_FACTOR": {
            "desc": "Overlap factor (0.0 - <1.0). High overlap (0.9) required for smooth tracking."
        },
        "KALMAN_PROCESS_NOISE": {
            "desc": "Process noise (Q). System volatility. High = Adapts fast (jittery); Low = Stiff model (smooth)."
        },
        "KALMAN_MEASUREMENT_NOISE": {
            "desc": "Measurement noise (R). Sensor uncertainty. High = Trust model (smooth); Low = Trust data (noisy)."
        },
        "MODAL_POWER_THRESHOLD": {
            "desc": "Min normalized power for frequency peak. Cutoff (0.0-1.0). Low = Detects weak signals/noise; High = Only strong peaks."
        },
        "TOMO_Z_MIN_M": {"desc": "Min altitude/depth (m)."}, "TOMO_Z_MAX_M": {"desc": "Max altitude/depth (m)."},
        "APPLY_LOG_SCALING": {
            "options": "[True, False]", 
            "desc": "Log scale input data. Compresses dynamic range to show weak internal reflections."
        },
        "APPLY_WINDOWING": {
            "options": "[True, False]", 
            "desc": "Apply Hanning window. Tapers signal edges to prevent FFT spectral leakage."
        },
        "APPLY_AUTOFOCUS": {
            "options": "[True, False]", 
            "desc": "Run autofocus algorithm."
        },
        "AUTOFOCUS_ITERATIONS": {
            "desc": "Autofocus iterations. Number of phase refinement loops. Standard=3. Too many can cause artifacts."
        },
        "APPLY_SPATIAL_FILTER": {
            "options": "[True, False]", 
            "desc": "Spatial smoothing. Blurs pixel-to-pixel noise."
        },
        "SPATIAL_FILTER_SIZE": {
            "desc": "Kernel size (px). Smoothing width. 3=Subtle; 9=Strong Blur. Large kernels kill fine details."
        },
        "APPLY_KALMAN_FILTER": {
            "options": "[True, False]", 
            "desc": "Kalman filtering. Tracks motion over time to reject random temporal jitter."
        },
        "APPLY_SVD_FILTER": {
            "options": "[True, False]", 
            "desc": "SVD filtering (clutter removal). Removes static background via Singular Value Decomposition."
        },
        "SVD_NUM_COMPONENTS": {
            "desc": "Components to remove. Number of dominant layers to cut (1=Background). Removing >1 may remove signal."
        },
        "COMPUTE_BASELINE_TOMOGRAM": {
            "options": "[True, False]", 
            "desc": "Compute baseline? Generate standard Beamforming reference alongside advanced method."
        },
        "SEISMIC_DAMPING_COEFF": {
            "desc": "Damping coefficient. Stabilizes inversion matrix. Low (0.01) = noisy; High (0.5) = blurry."
        },
        "PERFORM_MODAL_ANALYSIS": {
            "options": "[True, False]", 
            "desc": "Run Modal Analysis? Pre-scans data to identify dominant vibration frequencies."
        },
        "UPSAMPLE_FACTOR": {
            "desc": "Sub-pixel tracking precision. Factor to subdivide pixels (1200x ~ microns). Higher = Slower but precise."
        }
    }

    def get_typed_value(key, str_value, default_value):
        if isinstance(default_value, bool): return str_value.lower() in ['true', 't', '1', 'yes', 'y']
        if isinstance(default_value, int): return int(str_value)
        if isinstance(default_value, float): return float(str_value)
        return str_value

    params = defaults.copy()
    if 'user_parameters' in full_config:
        saved_params = full_config['user_parameters']
        saved_params_ci = {k.upper(): v for k, v in saved_params.items()}
        for key in param_keys:
            if key in saved_params_ci:
                params[key] = get_typed_value(key, saved_params_ci[key], params[key])

    print("\n   Current Settings:")
    print("-" * 50)
    for key in param_keys:
        if key in params:
            val = params[key]
            if key in explanations and "map" in explanations[key]:
                mapping = explanations[key]["map"]
                for k_map, v_map in mapping.items():
                    if str(v_map).lower() == str(val).lower(): val = f"{val} ({k_map})"; break
            print(f"  {key:<30}: {val}")
    print("-" * 50)

    # --- V7-29 RESUME LOCK LOGIC ---
    if lock_mode:
        print("\n[RESUME LOCKED] Parameters fixed to preserve scientific consistency.")
        print("To change settings, you must start a FRESH run (select 'n' at Resume prompt).")
        return params

    print("\nNOTE: You can accept defaults by pressing Enter.")
    if input("Do you want to change any of these parameters? (y/N): ").lower() in ['y', 'yes']:
        for key in param_keys:
            if key not in params: continue
            orig = params[key]
            expl = explanations.get(key, {})
            print("-" * 20, flush=True)
            print(f"Parameter: {key}")
            if "desc" in expl: print(f"Description: {expl['desc']}")
            if "table" in expl: 
                print("\nReference Velocities:"); 
                for l in expl['table']: print(l)
            if "details" in expl:
                for l in expl['details']: print(l)
            if "map" in expl:
                print(f"Options: {expl['options_text']}")
                curr_num = "?"
                for k_map, v_map in expl["map"].items():
                    if str(v_map).lower() == str(orig).lower(): curr_num = k_map; break
                new_v = input(f"Enter option number [Default: {orig} ({curr_num})]: ")
                if new_v:
                    try: 
                        if int(new_v) in expl["map"]: params[key] = expl["map"][int(new_v)]; print(f"   -> Set to: {params[key]}")
                    except: print("   Invalid.")
            else:
                new_v = input(f"Enter new value [Default: {orig}]: ")
                if new_v:
                    if key == "INVESTIGATION_FREQUENCY_HZ" and new_v.lower() == 'auto': params[key] = 'auto'
                    else: 
                        try: params[key] = get_typed_value(key, new_v, orig)
                        except: pass
    return params

# =============================================================================
#   DEFAULT CONFIGURATION
# =============================================================================
CONFIG = {
    'TOMOGRAPHY_MODE': 'FixedVelocity', 'TARGET_TYPE': 'building',          
    'SEISMIC_VELOCITY_MS': 555.0, 'INVESTIGATION_FREQUENCY_HZ': 50.0, 
    'NUM_LOOKS': 1000, 'OVERLAP_FACTOR': 0.90, 'UPSAMPLE_FACTOR': 1200,            
    'MICROMOTION_WINDOW': 64, 'APPLY_SVD_FILTER': True, 'SVD_NUM_COMPONENTS': 1,            
    'APPLY_KALMAN_FILTER': True, 'KALMAN_PROCESS_NOISE': 0.01, 'KALMAN_MEASUREMENT_NOISE': 0.1,    
    'FINAL_METHOD': 'cs', 'CS_NOISE_EPSILON': 0.1,
    'TOMO_Z_MIN_M': -10, 'TOMO_Z_MAX_M': 100,
    'AUTO_FREQ_RANGE': (1.0, 150.0), 'AUTO_VEL_RANGE': (300.0, 1000.0), 'AUTO_STEPS': 30,
    'APPLY_AUTOFOCUS': False, 'AUTOFOCUS_ITERATIONS': 3, 'APPLY_LOG_SCALING': True,
    'APPLY_WINDOWING': True, 'APPLY_SPATIAL_FILTER': True, 'SPATIAL_FILTER_SIZE': 5,
    'SEISMIC_DAMPING_COEFF': 0.15, 'PERFORM_MODAL_ANALYSIS': True, 'MODAL_POWER_THRESHOLD': 0.2,
    'RUN_ALL_FOCUSING_METHODS': False, 'COMPUTE_BASELINE_TOMOGRAM': True,
    'V_MIN_MS': 300.0, 'V_MAX_MS': 1000.0, 'V_STEPS': 20
}

def main():
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n--- 3D Seismic Tomography Processor V7-41 (Fast Probe Fix) ---")
    
    # --- 0. STORAGE & RESUME CHECK ---
    print("\n[STORAGE] Where do you want to save temporary files? (Recommended: External HDD)")
    print("Press Enter to use current folder, or paste a path (e.g., E:\\SAR_Work_Dir)")
    base_storage_dir = input("Path > ").strip()
    if base_storage_dir == "": base_storage_dir = os.getcwd()
    elif not os.path.exists(base_storage_dir):
        try: os.makedirs(base_storage_dir); print(f"[STORAGE] Created directory: {base_storage_dir}")
        except: base_storage_dir = os.getcwd()
            
    print(f"[STORAGE] Working Directory: {base_storage_dir}")
    
    search_pattern = os.path.join(base_storage_dir, "temp_results_*")
    existing_temps = sorted(glob.glob(search_pattern))
    
    resume_mode = False
    skipped_cols = set()
    temp_dir = os.path.join(base_storage_dir, f"temp_results_{run_timestamp}")
    selection_state_file = None
    
    if existing_temps:
        latest_temp = existing_temps[-1]
        folder_name = os.path.basename(latest_temp)
        print(f"\n[RECOVERY] Found existing process folder: {latest_temp}")
        ans = input(f"Do you want to RESUME this run? (y/n) [y]: ").strip().lower()
        if ans == '' or ans == 'y':
            temp_dir = latest_temp
            resume_mode = True
            
            # Check for Smart Selection Memory
            pot_state_file = os.path.join(temp_dir, 'selection_state.npz')
            if os.path.exists(pot_state_file):
                selection_state_file = pot_state_file
                print(f"[SMART RESUME] Found saved selection state. Skipping Map GUI.")
            
            processed_files = glob.glob(os.path.join(temp_dir, "slice_*.npz"))
            for pf in processed_files:
                try: skipped_cols.add(int(re.search(r'slice_(\d+).npz', pf).group(1)))
                except: pass
            print(f"[RECOVERY] Resuming. Found {len(skipped_cols)} already processed columns.")
            try: run_timestamp = folder_name.replace("temp_results_", "")
            except: pass
    
    output_filename = f"tomography_3D_results_{run_timestamp}.npz"
    print(f"Final output will be saved to local project folder: {output_filename}", flush=True)
    
    if not resume_mode:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

    try:
        # --- 1. FILE FINDING ---
        full_config, _ = load_config()
        if not full_config: full_config = {}
        
        file_paths = get_external_data_paths()
        if not file_paths:
            tif_files = glob.glob("*.tif")
            if tif_files: file_paths = {'tiff_file': os.path.abspath(tif_files[0])}
            else: print("CRITICAL: No .tif files found."); return
        
        base_name = os.path.splitext(file_paths['tiff_file'])[0]
        parent_dir = os.path.dirname(file_paths['tiff_file'])
        txt_files = glob.glob(os.path.join(parent_dir, f'{os.path.basename(base_name)}_metadata.txt'))
        if not txt_files: txt_files = glob.glob(os.path.join(parent_dir, "*.txt"))
        if txt_files: file_paths['txt_file'] = txt_files[0]
        else: file_paths['txt_file'] = None

        # --- 2. SMART RESUME & SANDBOX LOADING ---
        complex_data_full = None
        radar_params = None
        user_selection = None
        
        if resume_mode and selection_state_file:
            print(f"\n--- Smart Resume: Reloading data using saved selection... ---")
            try:
                state = np.load(selection_state_file, allow_pickle=True)
                if 'user_selection' in state:
                    raw_sel = state['user_selection']
                    if isinstance(raw_sel, np.ndarray) and raw_sel.ndim == 0:
                        user_selection = raw_sel.item()
                    else:
                        user_selection = raw_sel
                
                center_row = int(state['center_row'])
                central_col_idx = int(state['central_col_idx'])
                
                saved_radar_params = {}
                if 'radar_params' in state:
                    raw_rp = state['radar_params']
                    if isinstance(raw_rp, np.ndarray) and raw_rp.ndim == 0:
                        saved_radar_params = raw_rp.item()
                    else:
                        saved_radar_params = raw_rp
                
                scientific_metadata = {}
                ext_json_path = file_paths['tiff_file'].replace(".tif", "_extended.json")
                if not os.path.exists(ext_json_path):
                     ext_json_path = os.path.join(parent_dir, f"{os.path.basename(base_name)}_extended.json")

                if os.path.exists(ext_json_path):
                    try: 
                        scientific_metadata = parse_radar_parameters(ext_json_path)
                    except: pass
                
                if (not scientific_metadata or 'effective_prf' not in scientific_metadata) and os.path.exists(ext_json_path):
                    try:
                        with open(ext_json_path, 'r') as f:
                            raw_data = json.load(f)
                            found_prf = find_key_recursive(raw_data, 'prf')
                            if found_prf:
                                if isinstance(found_prf, dict):
                                    scientific_metadata['effective_prf'] = float(found_prf.get('mean', 5000.0))
                                else:
                                    scientific_metadata['effective_prf'] = float(found_prf)
                            found_bw = find_key_recursive(raw_data, 'bandwidth')
                            if found_bw:
                                scientific_metadata['doppler_bandwidth_hz'] = float(found_bw)
                    except: pass

                print(f"   Reloading massive TIF file... (Please wait)")
                loaded_data = load_mlc_data(file_paths['tiff_file'], {}) 
                if isinstance(loaded_data, tuple):
                    complex_data_full = loaded_data[0]
                    geometry_metadata = loaded_data[1]
                else:
                    complex_data_full = loaded_data
                    geometry_metadata = {}

                radar_params = geometry_metadata.copy() if geometry_metadata else {}
                if scientific_metadata: 
                    clean_physics = {k: v for k, v in scientific_metadata.items() if v is not None}
                    radar_params.update(clean_physics)
                if saved_radar_params:
                    for k, v in saved_radar_params.items():
                        if v is not None:
                            radar_params[k] = v

                if user_selection and 'corners' in user_selection:
                    corners = user_selection['corners']
                    if corners['ul'][0] is not None:
                        radar_params['lat_upper_left'] = corners['ul'][0]
                        radar_params['lon_upper_left'] = corners['ul'][1]
                        radar_params['lat_upper_right'] = corners['ur'][0]
                        radar_params['lon_upper_right'] = corners['ur'][1]
                        radar_params['lat_lower_left'] = corners['ll'][0]
                        radar_params['lon_lower_left'] = corners['ll'][1]
                        radar_params['lat_lower_right'] = corners['lr'][0]
                        radar_params['lon_lower_right'] = corners['lr'][1]

                if 'effective_prf' not in radar_params and 'prf_stats' not in radar_params:
                    radar_params['effective_prf'] = 5000.0
                    radar_params['doppler_bandwidth_hz'] = 500e6 
                    print("   [SAFE MODE] Physics parameters defaulted (PRF=5000Hz).")

                print("   Data reload complete.")
                
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[WARNING] Smart Resume failed. Falling back to manual selection.")
                resume_mode = False 
        
        if complex_data_full is None:
            print("\n--- Image Selection ---")
            print(f"[ACTION REQUIRED] A Map Window will open shortly. Check Taskbar!")
            try:
                complex_data_full, radar_params, user_selection, center_row, central_col_idx = run_image_selection_pipeline(
                    file_paths, full_config, run_timestamp
                )
                if complex_data_full is not None:
                    state_save_path = os.path.join(temp_dir, 'selection_state.npz')
                    np.savez(state_save_path, user_selection=user_selection, center_row=center_row, 
                             central_col_idx=central_col_idx, radar_params=radar_params)
                    
                    try:
                        if 'column_ranges' not in full_config: full_config['column_ranges'] = {}
                        full_config['column_ranges']['start_left'] = user_selection['start_left']
                        full_config['column_ranges']['end_left'] = user_selection['end_left']
                        full_config['column_ranges']['start_right'] = user_selection['start_right']
                        full_config['column_ranges']['end_right'] = user_selection['end_right']
                        if 'last_coordinates' not in full_config: full_config['last_coordinates'] = {}
                        full_config['last_coordinates']['lat'] = user_selection['lat_center']
                        full_config['last_coordinates']['lon'] = user_selection['lon_center']
                        if 'user_parameters' not in full_config: full_config['user_parameters'] = {}
                        full_config['user_parameters']['ANALYSIS_EXTENT_KM'] = user_selection['analysis_extent_km']
                        save_config(full_config)
                    except: pass
            except Exception as e:
                print(f"\n[ERROR] Selection Failed: {e}"); return
        
        if complex_data_full is None: return

        # --- 3. PARAMETERS ---
        initial_params_state = CONFIG.copy()
        if 'user_parameters' in full_config:
            for k, v in full_config['user_parameters'].items():
                if k in initial_params_state: 
                    try:
                        if isinstance(initial_params_state[k], bool): initial_params_state[k] = (v.lower() == 'true')
                        elif isinstance(initial_params_state[k], float): initial_params_state[k] = float(v)
                        elif isinstance(initial_params_state[k], int): initial_params_state[k] = int(v)
                        else: initial_params_state[k] = v
                    except: pass

        params = initial_params_state.copy()
        params['ANALYSIS_EXTENT_KM'] = user_selection['analysis_extent_km']
        param_order = [
            "TOMOGRAPHY_MODE", "TARGET_TYPE", 
            "SEISMIC_VELOCITY_MS", "V_MIN_MS", "V_MAX_MS", "V_STEPS",
            "INVESTIGATION_FREQUENCY_HZ", 
            "NUM_LOOKS", "OVERLAP_FACTOR", "UPSAMPLE_FACTOR",
            "FINAL_METHOD", "RUN_ALL_FOCUSING_METHODS", 
            "TOMO_Z_MIN_M", "TOMO_Z_MAX_M", "CS_NOISE_EPSILON", 
            "SEISMIC_DAMPING_COEFF", 
            "APPLY_LOG_SCALING", "APPLY_WINDOWING", 
            "PERFORM_MODAL_ANALYSIS", "MODAL_POWER_THRESHOLD", 
            "APPLY_SPATIAL_FILTER", "SPATIAL_FILTER_SIZE", 
            "APPLY_KALMAN_FILTER", "KALMAN_PROCESS_NOISE", "KALMAN_MEASUREMENT_NOISE", 
            "APPLY_SVD_FILTER", "SVD_NUM_COMPONENTS", 
            "APPLY_AUTOFOCUS", "AUTOFOCUS_ITERATIONS", 
            "COMPUTE_BASELINE_TOMOGRAM"
        ]

        while True:
            params = get_interactive_parameters(params, "Seismic Tomography Parameters", param_order, full_config, lock_mode=resume_mode)
            if resume_mode: break
            print(f"\n--- Saving selections... ---")
            full_config['user_parameters'].update({k: str(v) for k, v in params.items()})
            save_config(full_config)
            print("\n---------------------------------------------------------")
            confirm = input("Are all settings correct? (y/n) [y]: ").strip().lower()
            if confirm == '' or confirm == 'y': break

        # --- PRIORITY RECALL ---
        calib_file_path = os.path.join(temp_dir, 'calibrated_params.json')
        if os.path.exists(calib_file_path):
            print(f"\n[RESUME PRIORITY] Found saved calibration file: {calib_file_path}")
            try:
                with open(calib_file_path, 'r') as f:
                    saved_calib = json.load(f)
                params.update(saved_calib)
                print(f"   [SYSTEM] Overwrote parameters with calibrated values.")
            except: pass
        
        # --- VERIFICATION TABLE ---
        print("\n" + "="*60)
        print("   FINAL RUN CONFIGURATION (VERIFIED)")
        print("="*60)
        print(f"   {'PARAMETER':<35} | {'VALUE'}")
        print("-" * 60)
        for key in sorted(params.keys()):
            val = params[key]
            if isinstance(val, (list, dict)): val = str(val)
            print(f"   {key:<35} | {val}")
        print("="*60 + "\n")

        # --- 4. PROCESSING SETUP ---
        complex_data = complex_data_full
        center_set = {central_col_idx}
        l_start, l_end = user_selection['start_left'], user_selection['end_left']
        left_range = list(range(central_col_idx - l_end, central_col_idx - l_start + 1))
        r_start, r_end = user_selection['start_right'], user_selection['end_right']
        right_range = list(range(central_col_idx + r_start, central_col_idx + r_end + 1))
        
        cols_to_process = sorted(list(set(left_range) | center_set | set(right_range)))
        cols_to_process = [c for c in cols_to_process if 0 <= c < complex_data.shape[1]]
        
        if not cols_to_process: return

        prf = radar_params.get('effective_prf')
        if prf is None and 'prf_stats' in radar_params: prf = radar_params['prf_stats'].get('mean')
        if prf is None: prf = 5000.0

        spacing_m = radar_params.get('azimuth_spacing_m', 1.0)
        extent_pixels = int(round((user_selection["analysis_extent_km"] * 1000) / spacing_m))
        start_pixel = max(0, center_row - extent_pixels)
        end_pixel = min(radar_params['scene_rows'] - 1, center_row + extent_pixels)
        
        if params.get("APPLY_AUTOFOCUS", False):
            complex_data = apply_autofocus(complex_data, params["AUTOFOCUS_ITERATIONS"])

        z_vec, processed_count = None, 0
        final_freq = params["INVESTIGATION_FREQUENCY_HZ"]
        final_vel = params["SEISMIC_VELOCITY_MS"]
        selected_target = params['TARGET_TYPE']
        
        # --- STATISTICAL CALIBRATION PROBE (FIXED SPEED) ---
        # FIX: Check if calibration file exists before skipping
        calib_file_path = os.path.join(temp_dir, 'calibrated_params.json')
        calibration_exists = os.path.exists(calib_file_path)

        # Run probe if NOT resuming OR if resuming but calibration is missing
        if not resume_mode or (resume_mode and not calibration_exists):
            if resume_mode and not calibration_exists:
                print("\n[RESUME WARNING] Calibration file missing. Re-running Probe to ensure safety.")
            
            print("\n" + "="*60)
            print("   PRE-FLIGHT CALIBRATION (STATISTICAL PROBE)")
            print("="*60)
            
            # Auto-select 'y' if resuming to avoid stalling, otherwise ask user
            prompt_text = "Run Multi-Column Calibration (20-point Probe) to auto-detect physics? (y/n) [y]: "
            if resume_mode:
                print(f"{prompt_text} y (Auto-selected for Resume)")
                ans = 'y'
            else:
                ans = input(prompt_text).strip().lower()

            if ans in ['', 'y']:
                
                num_probes = 20
                step = max(1, len(cols_to_process) // num_probes)
                probe_indices = cols_to_process[::step][:num_probes]
                
                print(f"   Running probe on {len(probe_indices)} representative columns...")
                print(f"   Indices: {probe_indices}")
                
                detected_freqs = []
                detected_vels = []
                
                for p_idx_count, p_col in enumerate(probe_indices):
                    if start_pixel >= complex_data.shape[0] or p_col >= complex_data.shape[1]: continue
                    t_line = complex_data[start_pixel:end_pixel+1, p_col]
                    if np.max(np.abs(t_line)) < 1e-9: continue
                    if params.get("APPLY_LOG_SCALING", False):
                        t_line = np.log1p(np.abs(t_line)) * np.exp(1j * np.angle(t_line))
                    
                    sub_ap_args = {
                        "num_looks": params["NUM_LOOKS"], 
                        "overlap_factor": params["OVERLAP_FACTOR"],
                        "doppler_ambiguity_spacing_hz": prf,
                        "quiet": True # Added quiet flag to suppress phase 2 prints during probe
                    }
                    if radar_params.get('doppler_params_available', False):
                        sub_ap_args.update({
                            "doppler_bandwidth_hz": radar_params.get('doppler_bandwidth_hz', 120000.0), 
                            "doppler_centroid_hz": radar_params.get('doppler_centroid_initial_hz', 0.0),
                            "sar_metadata": radar_params
                        })

                    try:
                        l_res, sa_centers = generate_sub_aperture_slcs(transform_to_frequency_domain(t_line), **sub_ap_args)
                        if len(l_res) < 2: continue
                        
                        # --- CRITICAL FIX: FORCE LOW PRECISION FOR PROBE ---
                        Y_probe = estimate_micro_motions_sliding_master(
                            l_res, 
                            window_size=params['MICROMOTION_WINDOW'], 
                            upsample_factor=20 # Force 20x for speed (ignore global 1200x)
                        )
                        
                        if Y_probe.size == 0: continue
                        
                        d_f = perform_frequency_sweep_snr(Y_probe, sa_centers[:Y_probe.shape[1]], radar_params, params['AUTO_FREQ_RANGE'], params['AUTO_STEPS'], params['SEISMIC_VELOCITY_MS'])
                        d_v = perform_velocity_autofocus(Y_probe, sa_centers[:Y_probe.shape[1]], radar_params, d_f, params['AUTO_VEL_RANGE'], params['AUTO_STEPS'])
                        
                        detected_freqs.append(d_f)
                        detected_vels.append(d_v)
                        print(f"   [Probe {p_idx_count+1}/{len(probe_indices)}] Col {p_col}: {d_f:.1f} Hz, {d_v:.0f} m/s")
                    except Exception as e:
                         print(f"   [Probe Failed] Col {p_col}: {e}")

                if detected_freqs:
                    med_freq = np.median(detected_freqs)
                    med_vel = np.median(detected_vels)
                    p_target = 'building' if med_vel < 1000 else ('geology' if med_vel < 4500 else 'bridge')
                    
                    print("\n   --- CALIBRATION RESULTS (MEDIAN) ---")
                    print(f"   Detected Frequency : {med_freq:.1f} Hz")
                    print(f"   Detected Velocity  : {med_vel:.0f} m/s")
                    print(f"   Suggested Target   : {p_target.upper()}")
                    
                    # If resuming, auto-accept. If fresh, ask.
                    confirm_calib = 'y'
                    if not resume_mode:
                        confirm_calib = input("\n   Apply these settings for the full run? (y/n) [y]: ").strip().lower()
                    else:
                        print(f"\n   [RESUME] Auto-applying calibration results.")

                    if confirm_calib in ['', 'y']:
                        final_freq = med_freq
                        final_vel = med_vel
                        selected_target = p_target
                        
                        params['INVESTIGATION_FREQUENCY_HZ'] = final_freq
                        params['SEISMIC_VELOCITY_MS'] = final_vel
                        params['TARGET_TYPE'] = selected_target
                        full_config['user_parameters'].update({k: str(v) for k, v in params.items()})
                        save_config(full_config)

                        calibration_data = {
                            'INVESTIGATION_FREQUENCY_HZ': final_freq,
                            'SEISMIC_VELOCITY_MS': final_vel,
                            'TARGET_TYPE': selected_target
                        }
                        
                        try:
                            with open(calib_file_path, 'w') as f:
                                json.dump(calibration_data, f, indent=4)
                            print(f"   [SYSTEM] Calibration committed to disk: {calib_file_path}")
                        except Exception as e:
                            print(f"   [WARNING] Failed to save calibration: {e}")

                        print("   ✓ Settings updated and saved.")
                    else:
                        print("   -> Keeping manual settings.")
                else:
                    print("   [WARNING] Probe failed to detect valid signals. Using manual settings.")
            else:
                 print("   Skipped calibration.")
        else:
            print(f"[RESUME] Found valid calibration file ({os.path.basename(calib_file_path)}). Skipping Probe.")

        try:
            for i, current_col_idx in enumerate(cols_to_process):
                if resume_mode and current_col_idx in skipped_cols: continue

                print(f"\n--- Processing Column {current_col_idx} ({i+1}/{len(cols_to_process)}) ---", flush=True)
                if start_pixel >= complex_data.shape[0] or current_col_idx >= complex_data.shape[1]: continue
                
                tomographic_line_raw = complex_data[start_pixel:end_pixel+1, current_col_idx]
                if np.max(np.abs(tomographic_line_raw)) < 1e-9: continue
                
                tomographic_line_main = np.copy(tomographic_line_raw)
                
                if params.get("APPLY_SPATIAL_FILTER", False):
                    f_size = params.get("SPATIAL_FILTER_SIZE", 5)
                    tomographic_line_main = np.convolve(tomographic_line_main, np.ones(f_size)/f_size, 'same')
                if params.get("APPLY_LOG_SCALING", False):
                    tomographic_line_main = np.log1p(np.abs(tomographic_line_main)) * np.exp(1j * np.angle(tomographic_line_main))

                sub_ap_args = {
                    "num_looks": params["NUM_LOOKS"], 
                    "overlap_factor": params["OVERLAP_FACTOR"],
                    "doppler_ambiguity_spacing_hz": prf
                }
                if radar_params.get('doppler_params_available', False):
                    sub_ap_args.update({
                        "doppler_bandwidth_hz": radar_params.get('doppler_bandwidth_hz', 120000.0), 
                        "doppler_centroid_hz": radar_params.get('doppler_centroid_initial_hz', 0.0),
                        "sar_metadata": radar_params
                    })
                
                low_res_slcs, sub_ap_centers = generate_sub_aperture_slcs(transform_to_frequency_domain(tomographic_line_main), **sub_ap_args)
                if len(low_res_slcs) < 2: continue

                checkpoint_file = os.path.join(temp_dir, f"checkpoint_col_{current_col_idx}.npz")
                Y_processed = estimate_micro_motions_sliding_master(
                    low_res_slcs, 
                    window_size=params['MICROMOTION_WINDOW'], 
                    upsample_factor=params['UPSAMPLE_FACTOR'],
                    checkpoint_path=checkpoint_file
                )
                
                if Y_processed.size == 0: continue
                
                num_looks_for_focusing = Y_processed.shape[1]
                sub_ap_centers = sub_ap_centers[:num_looks_for_focusing]
                
                if params.get("APPLY_KALMAN_FILTER", False): 
                    Y_processed = apply_kalman_filter(Y_processed, params["KALMAN_PROCESS_NOISE"], params["KALMAN_MEASUREMENT_NOISE"])
                if params.get("APPLY_SVD_FILTER", False):
                    Y_processed = apply_svd_filter(Y_processed, n_components=params.get("SVD_NUM_COMPONENTS", 1))

                focus_args = {
                    "seismic_velocity_ms": final_vel, "vibration_frequency_hz": final_freq, 
                    "apply_windowing": params.get("APPLY_WINDOWING", True), 
                    "z_min": params["TOMO_Z_MIN_M"], "z_max": params["TOMO_Z_MAX_M"], 
                    "epsilon": params["CS_NOISE_EPSILON"], "damping_coeff": params["SEISMIC_DAMPING_COEFF"],
                    "method": params["TOMOGRAPHY_MODE"], "final_method": params["FINAL_METHOD"],
                    "v_min": params["V_MIN_MS"], "v_max": params["V_MAX_MS"], "v_steps": params["V_STEPS"],
                    "target_type": selected_target,
                    "image_shape": tomographic_line_main.shape
                }
                
                tomo_complex = None
                velocity_map_2d = None
                robustness_score_2d = None

                if params["TOMOGRAPHY_MODE"] == 'FixedVelocity' and params.get("RUN_ALL_FOCUSING_METHODS", False):
                    methods = ['beamforming', 'capon', 'cs']
                    for m in methods:
                        run_args = focus_args.copy(); run_args['final_method'] = m
                        tsc, zc, _ = focus_sonic_tomogram(Y_processed, sub_ap_centers, radar_params, **run_args)
                        tomogram_stages[m.capitalize()] = np.abs(tsc)
                        if m.lower() == params["FINAL_METHOD"].lower(): tomo_complex, z_vec_current = tsc, zc
                    if tomo_complex is None: tomo_complex = tsc 
                else:
                    tomo_complex, z_vec_current, third_output = focus_sonic_tomogram(Y_processed, sub_ap_centers, radar_params, **focus_args)
                    tomo_final = np.abs(tomo_complex)
                    if params["TOMOGRAPHY_MODE"] == 'VelocitySpectrum': velocity_map_2d = third_output
                    elif params["TOMOGRAPHY_MODE"] == 'LayeredInversion': robustness_score_2d = third_output
                    tomogram_stages[params["TOMOGRAPHY_MODE"]] = tomo_final

                tomo_final = np.abs(tomo_complex)
                if z_vec is None: z_vec = z_vec_current
                coherence_map = calculate_coherence_map(Y_processed)
                
                freq_map, power_map, mode_map = None, None, None
                if params.get("PERFORM_MODAL_ANALYSIS", False): 
                    freq_map, power_map, mode_map = perform_modal_analysis(
                        Y_processed, 
                        params["SEISMIC_VELOCITY_MS"], 
                        prf, 
                        num_looks_for_focusing, 
                        params["MODAL_POWER_THRESHOLD"]
                    )

                slice_filename = os.path.join(temp_dir, f"slice_{current_col_idx}.npz")
                if velocity_map_2d is None: velocity_map_2d = np.zeros_like(tomo_final)
                if robustness_score_2d is None: robustness_score_2d = np.zeros_like(tomo_final)
                
                np.savez_compressed(slice_filename, tomogram_final=tomo_final, tomogram_complex=tomo_complex,
                                    tomogram_stages=tomogram_stages, coherence_map=coherence_map,
                                    frequency_map=freq_map, power_map=power_map, mode_number_map=mode_map,
                                    z_vec=z_vec, y_matrix_processed=Y_processed,
                                    velocity_map_2d=velocity_map_2d, robustness_score_2d=robustness_score_2d)
                processed_count += 1
                print(f"Saved slice_{current_col_idx}.npz to {temp_dir}")

        except KeyboardInterrupt:
            print("\n\n--- KEYBOARD INTERRUPT ---")
        
        if processed_count == 0: 
            print("No new columns processed in this session.")
            if not glob.glob(os.path.join(temp_dir, "slice_*.npz")):
                 print("No existing slices found either. Exiting.")
                 return
            else:
                 print("Proceeding to assembly with existing slices...")
        
        print(f"\n--- Assembling final 3D data cube... ---")
        tomogram_list, tomogram_complex_list = [], []
        stages_list, coherence_list = [], []
        freq_list, power_list, mode_list = [], [], []
        velocity_list, robustness_list, y_matrix_list = [], [], []
        processed_cols_final = []
        
        sorted_files = sorted(glob.glob(os.path.join(temp_dir, "slice_*.npz")), key=lambda f: int(re.search(r'slice_(\d+).npz', f).group(1)))
        
        target_shape = None
        for f in sorted_files:
            col_num = int(re.search(r'slice_(\d+).npz', f).group(1)); processed_cols_final.append(col_num)
            with np.load(f, allow_pickle=True) as d:
                t = d['tomogram_final']
                if target_shape is None: target_shape = t.shape
                if t.shape != target_shape:
                    res = np.zeros(target_shape, dtype=t.dtype)
                    r, c = min(t.shape[0], target_shape[0]), min(t.shape[1], target_shape[1])
                    res[:r, :c] = t[:r, :c]; t = res
                
                tomogram_list.append(t)
                if 'tomogram_complex' in d: tomogram_complex_list.append(d['tomogram_complex']) 
                if 'tomogram_stages' in d: stages_list.append(d['tomogram_stages'].item())
                coherence_list.append(d['coherence_map'])
                
                if 'frequency_map' in d and d['frequency_map'] is not None and d['frequency_map'].ndim > 0: freq_list.append(d['frequency_map'])
                if 'power_map' in d and d['power_map'] is not None: power_list.append(d['power_map'])
                if 'mode_number_map' in d and d['mode_number_map'] is not None: mode_list.append(d['mode_number_map'])
                
                if 'velocity_map_2d' in d and d['velocity_map_2d'] is not None: velocity_list.append(d['velocity_map_2d'])
                if 'robustness_score_2d' in d and d['robustness_score_2d'] is not None: robustness_list.append(d['robustness_score_2d'])
                if 'y_matrix_processed' in d: y_matrix_list.append(d['y_matrix_processed'])
                
                if z_vec is None: z_vec = d['z_vec']

        tomogram_cube_3d = np.transpose(np.stack(tomogram_list, axis=0), (1, 0, 2))
        
        if tomogram_complex_list: tomogram_cube_complex_3d = np.transpose(np.stack(tomogram_complex_list, axis=0), (1, 0, 2))
        else: tomogram_cube_complex_3d = np.zeros_like(tomogram_cube_3d, dtype=np.complex64)
        
        if coherence_list: 
            coherence_sheet = np.transpose(np.stack(coherence_list, axis=0), (1, 0))
            target_width = tomogram_cube_3d.shape[2]
            coherence_cube_3d = np.tile(coherence_sheet[:, :, np.newaxis], (1, 1, target_width))
        else: 
            coherence_cube_3d = np.zeros_like(tomogram_cube_3d, dtype=np.float32)

        frequency_map_2d = np.stack(freq_list, axis=0) if freq_list else np.zeros((len(processed_cols_final), tomogram_list[0].shape[0]), dtype=np.float32)
        power_map_2d = np.stack(power_list, axis=0) if power_list else np.zeros((len(processed_cols_final), tomogram_list[0].shape[0]), dtype=np.float32)
        mode_number_map_2d = np.stack(mode_list, axis=0) if mode_list else np.zeros((len(processed_cols_final), tomogram_list[0].shape[0]), dtype=np.float32)

        velocity_cube_3d = np.transpose(np.stack(velocity_list, axis=0), (1, 0, 2)) if velocity_list else None
        robustness_cube_3d = np.transpose(np.stack(robustness_list, axis=0), (1, 0, 2)) if robustness_list else None

        print("\n--- Finalizing Geographic Parameters ---")
        lats, lons = geocode_tomographic_line_coords(radar_params, tomogram_cube_3d.shape[0])
        
        range_spacing_m = radar_params.get('range_spacing_m', 1.0)
        extent_in_pixels_horizontal = int(round((user_selection['analysis_extent_km'] * 1000) / range_spacing_m))
        left_column = max(0, central_col_idx - extent_in_pixels_horizontal)
        right_column = min(radar_params['scene_cols'] - 1, central_col_idx + extent_in_pixels_horizontal)

        geo_start_top = get_pixel_geo_coord(start_pixel, left_column, radar_params)
        geo_end_bottom = get_pixel_geo_coord(end_pixel, right_column, radar_params)
        geo_left_center = get_pixel_geo_coord(center_row, left_column, radar_params)
        geo_right_center = get_pixel_geo_coord(center_row, right_column, radar_params)
        
        analysis_line_length_m = haversine_distance(geo_left_center[0], geo_left_center[1], geo_right_center[0], geo_right_center[1])
        
        dz = abs(z_vec[1] - z_vec[0]) if len(z_vec) > 1 else 1.0
        voxel_spacing_meters = np.array([dz, radar_params.get('azimuth_spacing_m', 1.0), radar_params.get('range_spacing_m', 1.0)])

        results_to_save = {
            'tomogram_cube': tomogram_cube_3d,
            'z_vec': z_vec,
            'radar_params': radar_params,
            'user_parameters': params,
            'tomography_mode': params['TOMOGRAPHY_MODE'],
            'detected_freq': final_freq, 'detected_vel': final_vel,
            'voxel_spacing_meters': voxel_spacing_meters,
            'geo_start': np.array(geo_start_top), 
            'geo_end': np.array(geo_end_bottom),
            'geo_left_center': np.array(geo_left_center), 
            'geo_right_center': np.array(geo_right_center),
            'analysis_line_length_m': analysis_line_length_m,
            'vertical_extent_m': abs(end_pixel - start_pixel) * radar_params.get('azimuth_spacing_m', 1.0),
            'horizontal_extent_m': abs(right_column - left_column) * radar_params.get('range_spacing_m', 1.0),
            'tomogram_cube_complex': tomogram_cube_complex_3d,
            'coherence_cube': coherence_cube_3d,
            'tomogram_stages_list': np.array(stages_list, dtype=object),
            'frequency_map_2d': frequency_map_2d,
            'power_map_2d': power_map_2d,
            'mode_number_map_2d': mode_number_map_2d,
            'velocity_map_3d': velocity_cube_3d,
            'robustness_score_3d': robustness_cube_3d,
            'y_matrix_processed_3d': np.array(y_matrix_list, dtype=object),
            'source_data_file': os.path.basename(file_paths['tiff_file']),
            'central_column_index': central_col_idx,
            'processed_column_indices': np.array(processed_cols_final),
            'velocity_cube': velocity_cube_3d, 
            'final_coherence_cube': coherence_cube_3d,
            'final_cube': tomogram_cube_3d,
            'final_velocity_cube': velocity_cube_3d,
            'z_vec_ref': z_vec,
            'all_cols': processed_cols_final,
            'vertical_pixel_range': np.array([start_pixel, end_pixel]),
            'x_ideal_1d': np.linspace(0, tomogram_cube_3d.shape[1], tomogram_cube_3d.shape[1]),
            'y_ideal_1d': np.linspace(0, tomogram_cube_3d.shape[0], tomogram_cube_3d.shape[0]),
            'pixel_rows': np.linspace(start_pixel, end_pixel, tomogram_cube_3d.shape[0], dtype=int),
            'is_pre_degraded': False,
            'ds_factor': 1,
            'source_file_type': 'main_3D_processor_V7'
        }

        print("\n[DEBUG] Verifying Output Data Types before Save...")
        required_cv5_keys = ['velocity_cube', 'final_coherence_cube', 'is_pre_degraded']
        for k in required_cv5_keys:
            if k not in results_to_save:
                print(f"   [CRITICAL ERROR] Missing CV5 key: '{k}'")
            else:
                print(f"   [OK] Found CV5 key: '{k}'")

        for key, val in results_to_save.items():
            if key.startswith('geo_') or key == 'radar_params':
                print(f"   Key: '{key}' | Type: {type(val)}")
                if isinstance(val, tuple):
                    print(f"     -> WARNING: '{key}' is a Tuple. It will be saved as a 0-d Object Array.")
        
        np.savez_compressed(output_filename, **results_to_save)
        print(f"✓ Results saved to {output_filename}")
        
        plt.figure(figsize=(10, 6))
        mid_slice = tomogram_list[len(tomogram_list)//2]
        extent = [0, mid_slice.shape[0], params['TOMO_Z_MIN_M'], params['TOMO_Z_MAX_M']]
        plt.imshow(mid_slice.T, aspect='auto', cmap='jet', origin='lower', extent=extent)
        plt.colorbar(label='Reflectivity')
        plt.title(f"Center Slice Tomogram\nTarget: {selected_target.upper()}")
        plt.tight_layout(); plt.show()

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n[CRASH PROTECTION] The temporary folder '{temp_dir}' has been PRESERVED.")
        print(f"To resume, restart this script and select 'y' when prompted to resume.")

if __name__ == "__main__":
    main()