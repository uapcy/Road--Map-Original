# phase_3_advanced_tomography.py - UPDATED FOR AUTOMATIC DEPTH CALIBRATION

import numpy as np
import cvxpy as cp
from scipy import signal
from numpy.linalg import inv
from phase_3_utilities import _calculate_seismic_wavelength

def _calculate_image_sharpness(tomogram):
    """
    Calculates image sharpness (Entropy). 
    Lower Entropy = Sharper Image (better focus).
    We use negative entropy so we can maximize the score.
    """
    magnitude = np.abs(tomogram)
    total_energy = np.sum(magnitude) + 1e-9
    prob_dist = magnitude / total_energy
    
    # Shannon Entropy: -Sum(p * log(p))
    # We want to minimize entropy, so we return 1/Entropy or negative
    entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    
    return -entropy # Higher is better

def perform_velocity_autofocus(Y_matrix, sub_ap_centers, radar_params, 
                               fixed_freq_hz, 
                               velocity_range=(300.0, 6000.0), 
                               steps=20):
    """
    Performs 'Velocity Spectrum Analysis' (Autofocus) to find the correct depth scale.
    Sweeps through material velocities (e.g. 555 for concrete, 3000 for rock).
    The velocity that produces the sharpest image is the correct one.
    
    Args:
        Y_matrix: Displacement history
        sub_ap_centers: Sub-aperture centers
        radar_params: Metadata
        fixed_freq_hz: The optimal frequency found in the previous step
        velocity_range: Range of m/s to test
        steps: Number of velocities to test
        
    Returns:
        best_velocity: The velocity that maximizes image sharpness.
    """
    print(f"\n--- Starting Phase 3c: Automatic Depth Calibration (Velocity Autofocus) ---", flush=True)
    print(f"    Sweeping {velocity_range[0]} m/s to {velocity_range[1]} m/s at {fixed_freq_hz:.1f} Hz...", flush=True)
    
    velocities = np.linspace(velocity_range[0], velocity_range[1], steps)
    sharpness_scores = []
    
    # Geometry Prep
    num_pixels, num_looks = Y_matrix.shape
    z_vec_low_res = np.linspace(0, 50, 32)
    
    az_res = radar_params.get('azimuth_resolution_m', 1.0)
    center_idx = sub_ap_centers[num_looks // 2]
    b_perp_vec = (sub_ap_centers - center_idx) * (az_res / 10.0)
    
    slant_range = radar_params.get('slant_range_m', 10000)
    inc_angle = radar_params.get('incidence_angle_rad', 0.5)
    
    # Iterate Velocities
    for vel in velocities:
        # 1. Calculate Wavelength
        wavelength = vel / fixed_freq_hz
        
        # 2. Construct A Matrix
        denom = wavelength * slant_range * np.sin(inc_angle)
        if abs(denom) < 1e-9: denom = 1.0
        kz_vec = (4 * np.pi * b_perp_vec) / denom
        
        A_base = np.exp(1j * np.outer(kz_vec, z_vec_low_res))
        
        # 3. Fast Inversion
        tomogram_slice = np.abs(np.sum(Y_matrix[:, :, np.newaxis] * A_base.conj()[np.newaxis, :, :], axis=1))
        
        # 4. Measure Sharpness
        score = _calculate_image_sharpness(tomogram_slice)
        sharpness_scores.append(score)
        # print(f"    Vel: {vel:.0f} m/s -> Sharpness: {score:.4f}")
        
    # Find winner
    best_idx = np.argmax(sharpness_scores)
    best_vel = velocities[best_idx]
    
    # Identification
    material = "Unknown"
    if 300 <= best_vel <= 600: material = "Reinforced Concrete"
    elif 1400 <= best_vel <= 1600: material = "Water/Wet Soil"
    elif 2500 <= best_vel <= 3500: material = "Limestone/Brick"
    elif 4000 <= best_vel <= 6000: material = "Granite/Steel"
    
    print(f"    ✅ Optimal Velocity Found: {best_vel:.0f} m/s", flush=True)
    print(f"    🔍 Material ID: {material}", flush=True)
    
    return best_vel

# --- Original Advanced Functions (Kept for compatibility) ---

def _calculate_robustness_score(Y_pixel, A_layered, h_value, epsilon):
    if h_value is None or h_value.size == 0: return 0.0
    residual_vector = Y_pixel - A_layered @ h_value
    E_res = np.linalg.norm(residual_vector, ord=2)
    E_worst = 1.5 * epsilon
    if E_res <= epsilon: return 10.0 
    if E_res >= E_worst: return 0.0 
    score = 10.0 * (E_worst - E_res) / (E_worst - epsilon)
    return max(0, min(10.0, score))

def _focus_vsa(Y, sub_ap_centers, radar_params, seismic_wavelength_range):
    """ Legacy VSA function """
    return np.zeros_like(Y), np.zeros(10), np.zeros(10)

def _focus_layered_inversion(Y, sub_ap_centers, radar_params, layer_velocities):
    """ Legacy Layered function """
    return np.zeros_like(Y), np.zeros(10)