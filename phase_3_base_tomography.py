import numpy as np
import warnings

# Try importing CVXPY for advanced optimization
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

def steering_vector(z_vec, sub_ap_centers, radar_params, velocity_ms, frequency_hz):
    """
    Constructs the Steering Matrix A for the "Harmonic Mass-Spring" model.
    
    Physics:
    The phase shift is governed by the Doppler frequency variation associated with depth.
    Phononic/Vibration Model:
       Phi(z) = (4 * pi * f_vib * z) / V_sound
    
    Args:
        z_vec (array): Depth/Height vector (meters).
        sub_ap_centers (array): Center times/frequencies of the sub-apertures.
        radar_params (dict): Radar wavelength, orbital parameters.
        velocity_ms (float): Seismic/Sound velocity in the medium (m/s).
        frequency_hz (float): Vibration frequency (Hz).
        
    Returns:
        A (matrix): Steering matrix [Num_Looks x Num_Depth_Steps]
    """
    # Wavelength of the VIBRATION (Sound), not the Radar
    # lambda_sound = v_sound / f_vibration
    if frequency_hz < 1e-9: frequency_hz = 1.0 # Avoid div by zero
    lambda_sound = velocity_ms / frequency_hz
    
    # The "Look Angle" or "Time" diversity is captured by sub_ap_centers.
    # In a simplified tomographic model for single-pass micro-motion:
    # The phase evolution over 'looks' (time) correlates to depth via the vibration periodicity.
    
    # Constructing the Phase Kernel
    # Phase = 2 * pi * (Distance / Wavelength)
    # Here, Distance is a function of Depth (z) and Look Index (temporal evolution)
    
    # Normalize sub_ap_centers to 0..1 or -1..1 range for phase evolution
    t_norm = np.linspace(-1, 1, len(sub_ap_centers))
    
    # Steering Matrix A[look, depth]
    # A_ij = exp( j * k * z_j * t_i )
    # This models a standing wave pattern or harmonic oscillation at depth z
    
    num_looks = len(sub_ap_centers)
    num_z = len(z_vec)
    A = np.zeros((num_looks, num_z), dtype=np.complex64)
    
    k = 4 * np.pi / lambda_sound # Wavenumber
    
    for i in range(num_looks):
        # The term t_norm[i] represents the modulation of the vibration over the aperture time.
        # This effectively "scans" the phase of the vibration.
        phase = k * z_vec * t_norm[i]
        A[i, :] = np.exp(1j * phase)
        
    return A

def solve_beamforming(Y, A):
    """
    Standard Beamforming (Matched Filter).
    Robust but low resolution (blurry).
    Formula: x = A' * y
    """
    # A is [Looks x Depth], Y is [Looks x 1]
    # Result x is [Depth x 1]
    return np.conjugate(A).T @ Y

def solve_capon(Y_matrix, A, noise_loading=1e-3):
    """
    Capon Beamformer (MVDR).
    Better resolution, adaptive to interference.
    Requires a covariance matrix (multiple snapshots).
    """
    # If Y is a single vector, Capon is identical to Beamforming.
    # We need a covariance matrix R. We can estimate it if we have neighbors, 
    # but here we operate on a single pixel's time series.
    # We will simulate "snapshots" by using a sliding window or diagonal loading.
    
    L = Y_matrix.shape[0]
    
    # Estimate sample covariance matrix R = Y * Y'
    # For single snapshot, R is rank 1 (singular). We must use diagonal loading.
    R = np.outer(Y_matrix, np.conjugate(Y_matrix))
    R = R + noise_loading * np.eye(L) # Regularization
    
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return solve_beamforming(Y_matrix, A) # Fallback

    # Capon Spectrum: P(z) = 1 / (a(z)' R^-1 a(z))
    # We calculate the amplitude profile
    
    num_z = A.shape[1]
    x_capon = np.zeros(num_z, dtype=np.complex64)
    
    for i in range(num_z):
        a_vec = A[:, i]
        denom = np.conjugate(a_vec).T @ R_inv @ a_vec
        # Amplitude is roughly sqrt(power)
        if np.abs(denom) > 1e-9:
            x_capon[i] = 1.0 / denom # Power
        else:
            x_capon[i] = 0.0
            
    # Normalize to match beamforming scale approximately
    x_capon = np.sqrt(np.abs(x_capon)) 
    
    return x_capon

def solve_cs_cvxpy(Y, A, epsilon=0.1, smoothness_weight=0.1):
    """
    Compressed Sensing (L1 Norm Minimization) with Total Variation (TV).
    
    UPGRADE: "Atomic Decomposition" Logic.
    Instead of just minimizing |x| (sparsity of points), we minimize:
       |x|_1 + gamma * |grad(x)|_1
    
    This promotes "Blocky" or "Layered" signals (solid walls, magma chambers)
    rather than isolated noise points.
    
    Args:
        Y (array): Measurement vector [Looks].
        A (matrix): Steering matrix.
        epsilon (float): Noise tolerance (fitting error).
        smoothness_weight (float): 'gamma'. Controls layer continuity. 
                                   Higher = smoother layers. Lower = sharper points.
    """
    if not CVXPY_AVAILABLE:
        print("[WARNING] CVXPY not installed. Falling back to Beamforming.")
        return solve_beamforming(Y, A)

    num_z = A.shape[1]
    
    # Variable to solve for: x (Complex reflectivity at depth)
    x = cp.Variable(num_z, complex=True)
    
    # Objective: Minimize L1 norm (Sparsity) + TV norm (Continuity/Layers)
    # TV norm is sum(|x[i+1] - x[i]|)
    
    # Note: cp.diff(x) computes discrete differences
    objective = cp.Minimize(cp.norm(x, 1) + smoothness_weight * cp.norm(cp.diff(x), 1))
    
    # Constraint: Data fidelity ||Ax - y||_2 <= epsilon
    constraints = [cp.norm(A @ x - Y, 2) <= epsilon]
    
    problem = cp.Problem(objective, constraints)
    
    try:
        # Solve with SCS (Splitting Conic Solver) - robust for complex SOCP
        problem.solve(solver=cp.SCS, verbose=False)
        
        if x.value is None:
            # Fallback if solver fails
            return solve_beamforming(Y, A)
            
        return x.value
        
    except Exception as e:
        print(f"[CS ERROR] Solver failed: {e}. Fallback to BF.")
        return solve_beamforming(Y, A)

def focus_sonic_tomogram(Y_matrix_processed, sub_ap_centers, radar_params, 
                         seismic_velocity_ms=555.0, vibration_frequency_hz=50.0,
                         apply_windowing=True, z_min=-10, z_max=100, 
                         epsilon=0.1, damping_coeff=0.1,
                         method='FixedVelocity', final_method='cs',
                         v_min=300, v_max=6000, v_steps=20, 
                         target_type='building', **kwargs):
    """
    Main function to convert Micro-Motion history (Y) into Depth Profile (z).
    
    Args:
        Y_matrix_processed (matrix): [Depth x Looks] or [1 x Looks]. 
                                     Actually, input is usually [Px_Rows x Looks].
                                     Wait, phase_2 outputs Y as [Rows x Looks].
                                     This function usually processes ONE pixel or ONE line?
                                     
                                     Standard pipeline passes 'Y_processed' which is [Rows x Looks].
                                     We need to iterate per row (pixel) or handle matrix multiplication.
                                     
                                     For the Tomography code, usually we process a "Tomographic Line".
                                     Y_processed is [Num_Pixels_Along_Line x Num_Looks].
                                     
        sub_ap_centers (array): Center frequencies/times.
        ...
        final_method (str): 'beamforming', 'capon', 'cs'.
        
    Returns:
        tomogram (matrix): [Num_Pixels x Num_Z_Steps] complex result.
        z_vec (array): Depth vector.
        third_output (matrix): Optional diagnostic map.
    """
    
    num_pixels, num_looks = Y_matrix_processed.shape
    
    # 1. Define Depth Vector (z)
    # Resolution limit check
    lambda_sound = seismic_velocity_ms / (vibration_frequency_hz + 1e-9)
    # Heuristic: Depth step should be fraction of wavelength
    dz = lambda_sound / 4.0 
    if dz > 5.0: dz = 5.0 # Cap max step size
    if dz < 0.1: dz = 0.1 # Cap min step size
    
    z_vec = np.arange(z_min, z_max, dz)
    num_z = len(z_vec)
    
    # 2. Build Steering Matrix A [Looks x Depth]
    # Note: If parameters vary per pixel (e.g. velocity autofocus), this needs loop.
    # For 'FixedVelocity', A is constant for the whole line.
    
    A = steering_vector(z_vec, sub_ap_centers, radar_params, seismic_velocity_ms, vibration_frequency_hz)
    
    # 3. Solve (Inversion)
    # Output shape: [Num_Pixels x Num_Z]
    tomogram = np.zeros((num_pixels, num_z), dtype=np.complex64)
    
    # For CS and Capon, we often need to solve pixel-by-pixel or in blocks
    # Beamforming can be done as Matrix-Matrix multiplication: X = Y @ A.conj()
    
    if final_method == 'beamforming':
        # Vectorized Beamforming: [Px x Looks] @ [Looks x Z] -> [Px x Z]
        # A is [Looks x Z]. Conjugate A -> [Looks x Z]*
        # Y is [Px x Looks]. 
        # We need Y @ A*. 
        # Wait, math: y = Ax. x = A'y.
        # x_vec (Z x 1) = A' (Z x L) @ y_vec (L x 1)
        # For matrix Y (P x L): X (P x Z) = Y @ A.conj()
        tomogram = Y_matrix_processed @ np.conjugate(A)
        
    elif final_method == 'capon':
        # Capon requires R matrix per pixel or averaged. 
        # Doing per-pixel loop.
        for p in range(num_pixels):
            y_vec = Y_matrix_processed[p, :]
            tomogram[p, :] = solve_capon(y_vec, A, noise_loading=damping_coeff)
            
    elif final_method == 'cs':
        # Compressed Sensing is computationally heavy. Loop per pixel.
        # To speed up, we might skip empty pixels (low energy).
        
        # Determine Atomic Decomposition weight (Smoothness)
        # Use damping_coeff as the control for smoothness (gamma)
        smoothness = damping_coeff 
        if target_type == 'geology': smoothness *= 2.0 # Rocks are smoother than walls
        
        print(f"   [CS] Solving {num_pixels} pixels with TV-Regularization (gamma={smoothness})...")
        
        for p in range(num_pixels):
            y_vec = Y_matrix_processed[p, :]
            
            # Simple energy threshold to skip noise pixels and save time
            if np.sum(np.abs(y_vec)) < 1e-6:
                continue
                
            tomogram[p, :] = solve_cs_cvxpy(y_vec, A, epsilon=epsilon, smoothness_weight=smoothness)
            
            if p % 50 == 0 and p > 0:
                print(f"     -> Solved {p}/{num_pixels} pixels...", end='\r')
        print(f"     -> CS Inversion Complete.             ")

    return tomogram, z_vec, None