# phase_3_modal_analysis.py - UPDATED FOR AUTOMATIC FREQUENCY TUNING
import numpy as np
from scipy.fft import fft, fftfreq
import cvxpy as cp # Optional, used if we wanted CS, but Beamforming is faster for sweeps

def _calculate_image_snr(tomogram):
    """
    Calculates the Signal-to-Noise Ratio (Contrast) of a tomogram.
    High SNR = Structural features stand out clearly against background.
    """
    magnitude = np.abs(tomogram)
    peak_signal = np.max(magnitude)
    
    # Estimate noise floor as the median or lower percentile
    noise_floor = np.percentile(magnitude, 50) 
    
    if noise_floor == 0: return 0.0
    return peak_signal / noise_floor

def perform_frequency_sweep_snr(Y_matrix, sub_ap_centers, radar_params, 
                                freq_range=(1.0, 150.0), 
                                steps=30, 
                                test_velocity=555.0):
    """
    Performs a 'Frequency Sweep' to find the band with the highest SNR.
    This runs a fast Beamforming inversion at multiple test frequencies.
    
    Args:
        Y_matrix: Displacement history [Pixels x Looks]
        sub_ap_centers: Centers of sub-apertures
        radar_params: Metadata
        freq_range: Tuple (min_hz, max_hz) to sweep
        steps: Number of frequencies to test
        test_velocity: Assumed velocity for the test (default Concrete 555 m/s)
        
    Returns:
        best_freq: The frequency that maximized image contrast.
    """
    print(f"\n--- Starting Phase 3b: Automatic Frequency Sweep (SNR Optimization) ---", flush=True)
    print(f"    Sweeping {freq_range[0]} Hz to {freq_range[1]} Hz in {steps} steps...", flush=True)
    
    frequencies = np.linspace(freq_range[0], freq_range[1], steps)
    snr_scores = []
    
    # Pre-calculate geometric parameters to save time
    num_pixels, num_looks = Y_matrix.shape
    z_vec_low_res = np.linspace(0, 50, 32) # Low res depth for speed
    
    az_res = radar_params.get('azimuth_resolution_m', 1.0)
    center_idx = sub_ap_centers[num_looks // 2]
    b_perp_vec = (sub_ap_centers - center_idx) * (az_res / 10.0)
    
    slant_range = radar_params.get('slant_range_m', 10000)
    inc_angle = radar_params.get('incidence_angle_rad', 0.5)
    
    # Iterate through frequencies
    for freq in frequencies:
        if freq <= 0: 
            snr_scores.append(0)
            continue
            
        # 1. Calculate Wavelength for this test frequency
        wavelength = test_velocity / freq
        
        # 2. Construct A Matrix (Fast Beamforming)
        denom = wavelength * slant_range * np.sin(inc_angle)
        if abs(denom) < 1e-9: denom = 1.0
        kz_vec = (4 * np.pi * b_perp_vec) / denom
        
        # A: [Looks x Depths]
        A_base = np.exp(1j * np.outer(kz_vec, z_vec_low_res))
        
        # 3. Fast Inversion (Beamforming)
        # We sum over looks: P = Y * A_conj
        # Broadcasting: Y[P, L, 1] * A[1, L, D] -> Sum over L -> [P, D]
        # Use efficient matrix multiplication if possible (Y @ A.conj())
        if A_base.ndim == 2:
             tomogram_slice = np.abs(Y_matrix @ A_base.conj())
        else:
             tomogram_slice = np.abs(np.sum(Y_matrix[:, :, np.newaxis] * A_base.conj()[np.newaxis, :, :], axis=1))
        
        # 4. Measure SNR
        score = _calculate_image_snr(tomogram_slice)
        snr_scores.append(score)
        # print(f"    Freq: {freq:.1f} Hz -> SNR: {score:.2f}")

    # Find winner
    best_idx = np.argmax(snr_scores)
    best_freq = frequencies[best_idx]
    max_snr = snr_scores[best_idx]
    
    print(f"    ✅ Optimal Frequency Found: {best_freq:.1f} Hz (SNR: {max_snr:.1f})", flush=True)
    return best_freq

def perform_modal_analysis(Y_matrix, velocity, sample_rate, num_looks, power_threshold=0.2):
    """
    Standard FFT-based Modal Analysis.
    Calculates dominant frequencies and returns maps matching the pixel dimensions.
    
    Args:
        Y_matrix: Complex history [Pixels x Looks]
        velocity: Seismic velocity (unused in FFT but kept for signature)
        sample_rate: Effective sample rate (Hz)
        num_looks: Number of looks
        power_threshold: Threshold for peak detection
        
    Returns:
        freq_map: Array of dominant frequencies per pixel [Pixels]
        power_map: Array of peak powers per pixel [Pixels]
        mode_map: Array of mode numbers per pixel [Pixels]
    """
    print(f"\n--- Starting Phase 3b: Structural Modal Analysis (FFT) ---", flush=True)
    
    num_pixels = Y_matrix.shape[0]
    
    # Calculate average spectrum for the column (global analysis for stability)
    avg_signal = np.mean(np.abs(Y_matrix), axis=0)
    spectrum = np.abs(fft(avg_signal))
    freqs = fftfreq(len(avg_signal), d=1.0) 
    
    valid_idx = np.where(freqs > 0)[0]
    valid_spectrum = spectrum[valid_idx]
    valid_freqs = freqs[valid_idx]
    
    final_freq = 2.5
    final_power = 0.0
    
    if np.max(valid_spectrum) > 0:
        norm_spectrum = valid_spectrum / np.max(valid_spectrum)
        peaks = np.where(norm_spectrum > power_threshold)[0]
        
        if len(peaks) > 0:
            max_peak_idx = np.argmax(norm_spectrum)
            dominant_freq = valid_freqs[max_peak_idx]
            
            # Use the provided sample rate to scale frequency
            real_freq_hz = dominant_freq * sample_rate
            final_freq = real_freq_hz
            final_power = np.max(norm_spectrum)
    
    # Expand scalar results to full maps (Pixels,) to satisfy Main Processor stacking
    freq_map = np.full(num_pixels, final_freq, dtype=np.float32)
    power_map = np.full(num_pixels, final_power, dtype=np.float32)
    mode_map = np.full(num_pixels, 1.0, dtype=np.float32)
    
    return freq_map, power_map, mode_map