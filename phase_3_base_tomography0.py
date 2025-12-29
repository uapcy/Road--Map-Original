# phase_3_base_tomography.py
"""
Phase 3: Tomographic Inversion Engine
Implements the inversion of the vibrational phasor into a spatial tomogram.
Supports Beamforming, Capon, and Compressed Sensing modes.
"""

import numpy as np
import cvxpy as cp
from scipy import signal
from numpy.linalg import inv
# Utilities for physics calculations
from phase_3_utilities import _calculate_seismic_wavelength, _safe_inverse, _calculate_tomographic_resolution

# -----------------------------------------------------------------------------
# 1. BEAMFORMING (Matched Filter) - Good for Bridges/Vibration
# -----------------------------------------------------------------------------
def _focus_beamforming(Y, A, apply_windowing):
    """ 
    Performs standard beamforming (matched filter) inversion.
    Mathematically equivalent to Pulse Compression.
    
    Args:
        Y: Data Matrix [Pixels x Looks]
        A: Steering Matrix [Pixels x Looks x Depths] OR [Looks x Depths]
        apply_windowing: Bool to apply Hanning window
    
    Returns:
        Tomogram [Pixels x Depths] (Complex)
    """
    print("    Method: Beamforming (Matched Filter)", flush=True)
    
    # Y is [Pixels x Looks]
    if apply_windowing:
        # Windowing in the 'Look' dimension (Time/Doppler)
        # Create window based on number of looks (columns of Y)
        window = np.hanning(Y.shape[1])
        # Apply window row-wise
        Y = Y * window
        
    # Standard Beamformer: P = A^H * y
    
    # OPTIMIZATION: Check if A is 2D (Constant Steering) or 3D (Variable Steering)
    if A.ndim == 2:
        # Memory Efficient Path: Matrix Multiplication
        # Y: [Pixels, Looks]
        # A: [Looks, Depths]
        # Result: [Pixels, Depths]
        # This avoids creating the massive (Pixels, Looks, Depths) intermediate array
        print("    [Optimization] Using efficient 2D Matrix Multiplication.", flush=True)
        tomogram_complex = Y @ A.conj()
        
    else:
        # Legacy/Variable Path: Element-wise Broadcast
        # A: [Pixels, Looks, Depths]
        # We expand Y to [Pixels, Looks, 1] for broadcasting
        print("    [Info] Using 3D Element-wise broadcasting.", flush=True)
        Y_expanded = Y[:, :, np.newaxis]
        # Sum over the 'Looks' dimension (axis 1)
        tomogram_complex = np.sum(Y_expanded * A.conj(), axis=1)
    
    return tomogram_complex

# -----------------------------------------------------------------------------
# 2. CAPON (MVDR) - Good for Geology/Volcanoes
# -----------------------------------------------------------------------------
def _focus_capon(Y, A, apply_windowing, covariance_window_size=7):
    """ 
    Performs Capon (Minimum Variance Distortionless Response) inversion.
    Excellent for continuous mediums like rock/magma.
    """
    print(f"    Method: Capon (MVDR) with window {covariance_window_size}", flush=True)
    num_pixels, num_looks = Y.shape
    
    # Handle A dimensions
    if A.ndim == 3:
        num_depths = A.shape[2] 
    else:
        num_depths = A.shape[1]

    tomogram = np.zeros((num_pixels, num_depths), dtype=np.complex64)
    
    # Add regularization to diagonal
    diagonal_loading = 1e-6 * np.eye(num_looks)
    
    for i in range(num_pixels):
        # Spatial averaging for covariance matrix R
        r_start = max(0, i - covariance_window_size // 2)
        r_end = min(num_pixels, i + covariance_window_size // 2 + 1)
        
        # Snapshots: [K_pixels, T_looks] -> Transpose to [T_looks, K_pixels]
        snapshots = Y[r_start:r_end, :].T 
        
        if snapshots.shape[1] < 1:
            continue
            
        # Calculate Sample Covariance Matrix R = E[y y^H]
        # Shape: [Looks x Looks]
        R = (snapshots @ snapshots.conj().T) / snapshots.shape[1]
        R += diagonal_loading
        
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
            
        # Capon Spectrum: P(z) = 1 / (a(z)^H * R^-1 * a(z))
        # Iterate over depths
        for k in range(num_depths):
            # Steering vector for this pixel and depth: [Looks]
            if A.ndim == 3:
                a_vec = A[i, :, k]
            else:
                a_vec = A[:, k]
            
            # Denominator = a^H R^-1 a
            denom = a_vec.conj().T @ R_inv @ a_vec
            
            # The result is Power (Real), but we return complex for consistency
            tomogram[i, k] = 1.0 / np.abs(denom) + 0j
            
    return tomogram

# -----------------------------------------------------------------------------
# 3. COMPRESSED SENSING (L1) - Good for Buildings/Echography
# -----------------------------------------------------------------------------
def _focus_compressed_sensing(Y, A, epsilon=0.1):
    """ 
    Performs L1-norm minimization.
    Ideal for 'sparse' targets like walls inside empty air.
    Note: Requires cvxpy. If too slow, falls back to Beamforming.
    """
    print("    Method: Compressed Sensing (L1)", flush=True)
    
    num_pixels, num_looks = Y.shape
    
    # Handle A dimensions
    if A.ndim == 3:
        num_depths = A.shape[2] 
    else:
        num_depths = A.shape[1]
        
    tomogram = np.zeros((num_pixels, num_depths), dtype=np.complex64)
    
    print("    Solving L1 minimization pixel-by-pixel (this may take time)...", flush=True)
    
    # We solve for a sparse reflectivity vector x at each pixel
    # minimize ||x||_1 subject to ||y - Ax||_2 <= epsilon
    
    for i in range(0, num_pixels, 10): # processing every 10th pixel for speed in this demo
        # In production, remove step or use batch solver
        y_vec = Y[i, :]
        
        if A.ndim == 3:
            A_mat = A[i, :, :] # [Looks x Depths]
        else:
            A_mat = A # [Looks x Depths]
        
        # Define variable (complex)
        x = cp.Variable(num_depths, complex=True)
        
        # Objective
        objective = cp.Minimize(cp.norm(x, 1))
        
        # Constraints
        constraints = [cp.norm(y_vec - A_mat @ x, 2) <= epsilon]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if x.value is not None:
                tomogram[i, :] = x.value
        except:
            # Fallback to matched filter if CS fails
            tomogram[i, :] = (y_vec @ A_mat.conj()).conj()

    return tomogram

# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
def focus_sonic_tomogram(Y_matrix, sub_ap_centers, radar_params, 
                         target_type='building', # 'building', 'geology', 'bridge'
                         vibration_frequency_hz=None, **kwargs):
    """
    Main function to focus the micro-motion history into a spatial image.
    Configures the physics engine based on Target Type.
    
    Args:
        Y_matrix: Complex displacement history [Pixels x Looks]
        sub_ap_centers: Center bins of sub-apertures
        radar_params: Dict of radar metadata
        target_type: Defines velocity model and solver
        vibration_frequency_hz: The resonant frequency to focus on
        **kwargs: absorbs extra arguments like 'image_shape', 'seismic_velocity_ms', 'final_method'
        
    Returns:
        tomogram: The focused image [Pixels x Depth]
        z_vec: The depth axis in meters
    """
    print(f"\n--- Starting Phase 3: Tomographic Inversion ({target_type.upper()}) ---", flush=True)
    
    # 1. Configuration based on Target Type and Overrides from kwargs
    if target_type == 'building':
        velocity = 555.0 # Speed of sound in Concrete
        method = 'cs'
    elif target_type == 'geology':
        velocity = 3000.0 # Speed of sound in Rock/Limestone
        method = 'capon'
    elif target_type == 'bridge':
        velocity = 5000.0 # Speed of sound in Steel
        method = 'beamforming' 
    else:
        velocity = 2000.0
        method = 'beamforming'

    # Override defaults if explicit values are passed (e.g., from Auto-Detection)
    if 'seismic_velocity_ms' in kwargs and kwargs['seismic_velocity_ms'] is not None:
        velocity = kwargs['seismic_velocity_ms']
    
    if 'final_method' in kwargs and kwargs['final_method'] is not None:
        method = kwargs['final_method']

    print(f"    Target: {target_type} -> Velocity: {velocity:.0f} m/s, Method: {method}", flush=True)
        
    # 2. Wavelength Calculation
    # If frequency is not provided, use physics defaults from papers
    if vibration_frequency_hz is None or vibration_frequency_hz <= 0:
        if target_type == 'bridge': 
            vibration_frequency_hz = 2.5 # Low freq resonance
        elif target_type == 'geology': 
            vibration_frequency_hz = 100.0 # Seismic noise
        else: 
            vibration_frequency_hz = 50.0 # Building hum
        print(f"    Using default frequency: {vibration_frequency_hz} Hz", flush=True)
        
    wavelength = _calculate_seismic_wavelength(velocity, vibration_frequency_hz)
    print(f"    Seismic Wavelength: {wavelength:.2f} m", flush=True)

    # 3. Construct Steering Matrix A
    # A(z) = exp(j * Kz * z)
    # Kz = 4 * pi * B_perp / (lambda * R * sin(theta))
    
    num_pixels, num_looks = Y_matrix.shape
    
    # Depth vector (0 to 100m into the target)
    z_vec = np.linspace(0, 100, 256) 
    num_depths = len(z_vec)
    
    # Calculate Virtual Baselines (B_perp)
    # The baseline is created by the satellite motion during the integration time.
    # B_perp is proportional to the separation between sub-apertures centers.
    
    az_res = radar_params.get('azimuth_resolution_m', 1.0)
    
    # Center index of the full aperture
    center_idx = sub_ap_centers[num_looks // 2]
    
    # Calculate displacement in meters for each look relative to center
    # Each bin in FFT corresponds to specific angle/time
    # This is a linear approximation valid for small angles
    b_perp_vec = (sub_ap_centers - center_idx) * (az_res / 10.0) # Scaling factor calibration
    
    # Pre-compute Kz for all looks
    slant_range = radar_params.get('slant_range_m', 10000)
    inc_angle = radar_params.get('incidence_angle_rad', 0.5)
    
    denom = wavelength * slant_range * np.sin(inc_angle)
    if abs(denom) < 1e-9: denom = 1.0
        
    kz_vec = (4 * np.pi * b_perp_vec) / denom
    
    # Build A Matrix: [Looks, Depths]
    # NOTE: We keep this as a 2D matrix for memory efficiency.
    # We do NOT tile it to (num_pixels, Looks, Depths) which causes MemoryError.
    
    A_base = np.exp(1j * np.outer(kz_vec, z_vec)) # [Looks, Depths]
    
    # 4. Execute Inversion
    # Pass A_base (2D) directly. The solvers are now optimized to handle it.
    
    if method == 'beamforming':
        tomogram = _focus_beamforming(Y_matrix, A_base, True)
    elif method == 'capon':
        tomogram = _focus_capon(Y_matrix, A_base, True)
    elif method == 'cs':
        # CS is computationally heavy, use with caution on large arrays
        tomogram = _focus_compressed_sensing(Y_matrix, A_base)
    else:
        tomogram = _focus_beamforming(Y_matrix, A_base, True)

    print("--- Tomography Complete ---", flush=True)
    return tomogram, z_vec, None