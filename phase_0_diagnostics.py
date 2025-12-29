# phase_0_diagnostics.py
# Performs "Pre-Flight" signal checks on the specific user-selected area.

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fft, ifft, ifftshift
import datetime
import time

def calculate_dynamic_range(magnitude_data):
    valid_data = magnitude_data[magnitude_data > 0]
    if len(valid_data) == 0: return 0.0
    peak_signal = np.percentile(valid_data, 99.9)
    noise_floor = np.percentile(valid_data, 10)
    if noise_floor == 0: return 0.0
    return 20 * np.log10(peak_signal / noise_floor)

def calculate_focus_score(complex_patch):
    f_transform = fftshift(fft2(complex_patch))
    magnitude_spectrum = np.abs(f_transform)
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8
    total_energy = np.sum(magnitude_spectrum)
    center_energy = np.sum(magnitude_spectrum[cy-radius:cy+radius, cx-radius:cx+radius])
    high_freq_energy = total_energy - center_energy
    return high_freq_energy / total_energy

def generate_diagnostic_stack(tomographic_line, num_looks=96, overlap=0.85):
    spectrum = fftshift(fft(tomographic_line))
    total_bandwidth = len(spectrum)
    look_bw = int(total_bandwidth / (num_looks * (1 - overlap) + overlap))
    step = int(look_bw * (1 - overlap))
    stack = []
    for i in range(num_looks):
        start = i * step
        end = start + look_bw
        if end > total_bandwidth: break
        sub_spectrum = spectrum[start:end] * np.hanning(end-start)
        padded = np.zeros(total_bandwidth, dtype=np.complex64)
        pad_start = (total_bandwidth - len(sub_spectrum)) // 2
        padded[pad_start : pad_start+len(sub_spectrum)] = sub_spectrum
        stack.append(ifft(ifftshift(padded)))
    return np.array(stack)

def run_diagnostic_check(complex_data, center_row, center_col):
    """
    Runs signal quality metrics on the specific area selected by the user.
    Returns True if user wants to proceed, False otherwise.
    """
    print(f"\n--- 🏥 PHASE 0: Diagnostic Check for Pixel ({center_row}, {center_col}) ---")
    
    # 1. Extract Patch (Static Analysis)
    rows, cols = complex_data.shape
    crop_size = 256
    r_start = max(0, center_row - crop_size//2)
    r_end = min(rows, center_row + crop_size//2)
    c_start = max(0, center_col - crop_size//2)
    c_end = min(cols, center_col + crop_size//2)
    
    patch = complex_data[r_start:r_end, c_start:c_end]
    
    # 2. Extract Line (Dynamic/Doppler Analysis)
    # We grab a vertical line 2048 pixels deep centered on the row
    line_len = 2048
    l_start = max(0, center_row - line_len//2)
    l_end = min(rows, center_row + line_len//2)
    probe_line = complex_data[l_start:l_end, center_col]
    
    # --- CALCULATIONS ---
    dr_db = calculate_dynamic_range(np.abs(patch))
    focus_score = calculate_focus_score(patch)
    
    stack = generate_diagnostic_stack(probe_line)
    
    # Coherence check
    coherences = []
    for i in range(len(stack) - 1):
        num = np.abs(np.sum(stack[i] * np.conj(stack[i+1])))
        den = np.sqrt(np.sum(np.abs(stack[i])**2) * np.sum(np.abs(stack[i+1])**2))
        if den > 0: coherences.append(num/den)
    avg_coherence = np.mean(coherences) if coherences else 0.0

    # --- REPORT ---
    print(f"1. DYNAMIC RANGE:    {dr_db:.1f} dB  ", end="")
    if dr_db > 15: print("(Good)")
    else: print("(Weak Contrast)")
    
    print(f"2. FOCUS SCORE:      {focus_score:.3f}   ", end="")
    if focus_score > 0.2: print("(Sharp)")
    else: print("(Blurry)")
    
    print(f"3. LOCAL COHERENCE:  {avg_coherence:.3f}   ", end="")
    if avg_coherence > 0.4: print("(Stable Signal)")
    elif avg_coherence > 0.2: print("(Noisy but Usable)")
    else: print("(Unstable/Noise)")

    # --- VISUALIZATION ---
    print("\nDisplaying diagnostic plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Static Image
    ax1.imshow(np.log1p(np.abs(patch)), cmap='gray')
    ax1.plot(center_col - c_start, center_row - r_start, 'rx', markersize=10) # Mark the spot
    ax1.set_title(f"Context Image (Target marked X)")
    
    # Doppler Stack
    stack_mag = np.abs(stack).T 
    ax2.imshow(np.log1p(stack_mag), aspect='auto', cmap='jet')
    ax2.set_title("Virtual Probe (Vibration Data)")
    ax2.set_xlabel("Time (Sub-Apertures)")
    ax2.set_ylabel("Depth (Pixels)")
    
    plt.tight_layout()
    
    # --- CORRECTED: DISPLAY WITH SAVE OPTION ---
    # Show non-blocking first so the window appears
    plt.show(block=False)
    plt.pause(0.5) # Allow time for the window to render on screen
    
    # Display diagnostic plots for 2 seconds
    print("Displaying diagnostic plots for 2 seconds...")
    plt.pause(2.0)
    
    # Ask user if they want to save the charts
    save_diag = input("\nDo you want to save the diagnostic charts? (y/n): ").lower().strip()
    if save_diag in ['y', 'yes']:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        diag_filename = f"diagnostic_charts_{ts}.png"
        fig.savefig(diag_filename, dpi=150)
        print(f"  ✓ Diagnostics saved as {diag_filename}")
    
    plt.close(fig)
    
    # --- AUTOMATIC PROCEED ---
    print("\n✅  Diagnostic check complete. Proceeding to full analysis...")
    return True