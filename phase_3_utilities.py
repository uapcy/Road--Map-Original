# phase_3_utilities.py

import numpy as np
from numpy.linalg import inv

# Set seismically realistic maximum velocity squared for clamping the Dix result
# V_max = 8000 m/s is a safe upper bound for crustal P-waves to prevent numerical blowup
V_MAX_SQ_CLAMP = 8000.0**2 

def _safe_inverse(matrix, regularization=1e-6):
    """Safe matrix inverse with regularization to avoid singular matrices."""
    try:
        return inv(matrix + regularization * np.eye(matrix.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix + regularization * np.eye(matrix.shape[0]))

def _calculate_seismic_wavelength(seismic_velocity_ms, vibration_frequency_hz):
    """Calculate seismic wavelength from velocity and frequency."""
    if vibration_frequency_hz <= 0:
        # Assuming a reasonable default velocity to prevent division by zero, 
        # as a non-positive frequency is usually an error in input configuration.
        return 0.1 # Placeholder: returning a very small wavelength to flag error
    return seismic_velocity_ms / vibration_frequency_hz

def _calculate_tomographic_resolution(seismic_wavelength, slant_range, orbital_aperture):
    """
    Calculate tomographic resolution according to paper equation.
    
    δ_T = (λ_seismic × R) / (2 × A)
    where A = orbital aperture
    """
    return (seismic_wavelength * slant_range) / (2 * orbital_aperture)
    
def _calculate_layered_velocity(v_nmo_sq, z_nmo, v_int_sq_acc_prev, z_prev):
    """
    Applies the discrete version of the Dix formula (V_int^2 = d(V_NMO^2 * T_NMO) / d(T_NMO)).
    Uses V_MAX_SQ_CLAMP constant for boundary checking.
    
    Args:
        v_nmo_sq (float): NMO velocity squared for the current depth slice.
        z_nmo (float): Current depth slice (proxy for T_NMO).
        v_int_sq_acc_prev (float): Accumulated (V_NMO^2 * T_NMO) from the previous depth slice.
        z_prev (float): Previous depth slice (proxy for T_prev).
        
    Returns:
        float: The calculated V_int^2 (interval velocity squared).
    """
    
    layer_thickness = z_nmo - z_prev
    
    if layer_thickness <= 1e-9:
        # If thickness is zero, assume same layer velocity
        return v_nmo_sq 
        
    # Total accumulated V_NMO^2 * T_NMO up to current depth
    v_nmo_sq_acc_curr = v_nmo_sq * z_nmo
    
    # Numerator: d(V_NMO^2 * T_NMO)
    numerator = v_nmo_sq_acc_curr - v_int_sq_acc_prev 
    
    # Dix Formula: V_int^2 = numerator / layer_thickness
    v_int_sq = numerator / layer_thickness
    
    # Apply stability checks and clamping
    v_int_sq = np.clip(v_int_sq, 1.0, V_MAX_SQ_CLAMP) # Clamp V_int^2 to 1.0 (min velocity of 1 m/s) and max clamp
    
    return v_int_sq