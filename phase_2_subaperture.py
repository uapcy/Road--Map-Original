# phase_2_subaperture.py
import numpy as np
from scipy.fft import ifft, fftshift, ifftshift

def generate_sub_aperture_slcs(tomographic_line_fft, num_looks=1000, overlap_factor=0.90,
                               doppler_bandwidth_hz=None, doppler_centroid_hz=0.0,
                               doppler_ambiguity_spacing_hz=None, max_unambiguous_doppler_hz=None,
                               sar_metadata=None, quiet=False):
    """
    Generate sub-aperture SLCs using the "Sliding Spectral Window" approach
    described in Biondi et al. (2022) for SAR Echography.
    
    Defaults are set to "Echography Quality":
    - num_looks: 1000 (creates a smooth time series)
    - overlap_factor: 0.90 (90% overlap for continuous motion tracking)
    """
    if not quiet:
        print(f"\n--- Starting Phase 2: Sub-Aperture Generation (Echography Mode) ---", flush=True)
        print(f"Targeting: {num_looks} looks with {overlap_factor*100:.1f}% overlap.", flush=True)
    
    azimuth_size = len(tomographic_line_fft)
    
    # --- PHYSICS-BASED DOPPLER BANDWIDTH CALCULATION ---
    def calculate_doppler_bandwidth_from_physics(metadata):
        try:
            if metadata is None: return None
            collect = metadata.get('collect', {})
            radar = collect.get('radar', {})
            antenna = collect.get('transmit_antenna', {})
            state = collect.get('state', {})
            
            # 1. Frequency
            center_freq = radar.get('center_frequency', 9.6e9)
            c = 299792458.0
            wavelength = c / center_freq
            
            # 2. Beamwidth
            az_beamwidth = antenna.get('azimuth_beamwidth', 0.005236)
            
            # 3. Velocity
            avg_vel = 7600.0
            if 'state_vectors' in state:
                vels = [sv.get('velocity', [0,0,0])[1] for sv in state['state_vectors']]
                if vels: avg_vel = np.mean(vels)
            
            return (2.0 * avg_vel * az_beamwidth) / wavelength
        except:
            return 22000.0 # Default X-band

    # Resolve Bandwidth
    if doppler_bandwidth_hz is None or doppler_bandwidth_hz <= 0:
        doppler_bandwidth_hz = calculate_doppler_bandwidth_from_physics(sar_metadata)
        if doppler_bandwidth_hz is None: doppler_bandwidth_hz = 22000.0
    
    # Resolve PRF
    prf = doppler_ambiguity_spacing_hz
    if prf is None and sar_metadata:
        try: prf = sar_metadata['collect']['radar']['prf']['prf']
        except: prf = 3000.0
    
    freq_res = prf / azimuth_size
    total_bins = int(doppler_bandwidth_hz / freq_res)
    
    # Calculate Window Size (Total Span = Window + (N-1)*Step)
    # Step = Window * (1-Overlap)
    denominator = 1 + (num_looks - 1) * (1 - overlap_factor)
    sub_ap_length_bins = int(total_bins / denominator)
    step_bins = int(sub_ap_length_bins * (1 - overlap_factor))
    
    if sub_ap_length_bins < 4: sub_ap_length_bins = 4
    if step_bins < 1: step_bins = 1
    
    if not quiet:
        print(f"[DOPPLER] Total BW: {doppler_bandwidth_hz:.0f} Hz ({total_bins} bins)", flush=True)
        print(f"[DOPPLER] Window: {sub_ap_length_bins} bins, Step: {step_bins} bins", flush=True)

    sub_aperture_slcs = []
    sub_ap_centers = []
    
    # Frequency axis
    freq_axis = np.fft.fftfreq(azimuth_size, d=1.0/prf)
    freq_axis = np.fft.fftshift(freq_axis)
    centroid_bin = np.argmin(np.abs(freq_axis - doppler_centroid_hz))

    # Generate Looks
    total_span_needed = sub_ap_length_bins + (num_looks - 1) * step_bins
    start_offset = centroid_bin - total_span_needed // 2
    
    for i in range(num_looks):
        start_bin = start_offset + i * step_bins
        
        # Circular wrap logic
        indices = np.arange(start_bin, start_bin + sub_ap_length_bins) % azimuth_size
        sub_ap_spectrum = tomographic_line_fft[indices]
        
        # Window & Zero Pad
        sub_ap_spectrum *= np.hanning(len(sub_ap_spectrum))
        padded = np.zeros(azimuth_size, dtype=np.complex64)
        center_pos = azimuth_size // 2 - len(sub_ap_spectrum) // 2
        padded[center_pos : center_pos + len(sub_ap_spectrum)] = sub_ap_spectrum
        
        sub_aperture_slcs.append(ifft(ifftshift(padded)))
        sub_ap_centers.append(start_bin + sub_ap_length_bins//2)

    return sub_aperture_slcs, np.array(sub_ap_centers)