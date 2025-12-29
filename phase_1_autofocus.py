# phase_1_autofocus.py - CORRECTED FOR LARGE DATASETS

import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift

def apply_autofocus(complex_data, iterations=3, chunk_size=256):  # Reduced iterations and chunk size
    """
    Applies Phase Gradient Autofocus in manageable chunks to handle large datasets.
    Fixed for very large arrays that cause memory issues.
    """
    print(f"\n--- Applying Phase Gradient Autofocus with {iterations} iterations ---", flush=True)
    if not np.iscomplexobj(complex_data):
        raise ValueError("Input data for autofocus must be complex.")

    num_rows, num_cols = complex_data.shape
    focused_data = np.zeros_like(complex_data, dtype=np.complex64)

    print(f"Processing {num_cols} columns...", flush=True)
    last_reported_col = 0

    for start_col in range(0, num_cols, chunk_size):
        # --- MODIFICATION: Add progress reporting every ~1000 columns ---
        if start_col - last_reported_col >= 1000:
            print(f"  Autofocus progress: Processed up to column {start_col} of {num_cols}...")
            last_reported_col = start_col

        end_col = min(start_col + chunk_size, num_cols)
        
        try:
            # Extract a chunk to process
            data_chunk = complex_data[:, start_col:end_col].copy()

            # Apply the iterative autofocus algorithm to this chunk
            for iter_num in range(iterations):
                # --- MODIFICATION: Removed per-iteration print statement ---
                
                # Transform to range-Doppler domain
                range_doppler_chunk = fftshift(fft(data_chunk, axis=0), axes=0)
                
                # Find the brightest pixel in each column (range bin)
                brightest_pixel_indices = np.argmax(np.abs(range_doppler_chunk), axis=0)
                
                # Center the data based on the brightest pixel
                chunk_rows, chunk_cols = range_doppler_chunk.shape
                centered_range_doppler = np.zeros_like(range_doppler_chunk)
                
                for col in range(chunk_cols):
                    shift = chunk_rows // 2 - brightest_pixel_indices[col]
                    # FIXED: Ensure shift is within valid range
                    if abs(shift) < chunk_rows:
                        centered_range_doppler[:, col] = np.roll(range_doppler_chunk[:, col], shift)
                    else:
                        centered_range_doppler[:, col] = range_doppler_chunk[:, col]
                
                # Estimate phase error from adjacent rows
                if chunk_rows > 1:  # Ensure we have at least 2 rows
                    phase_difference = centered_range_doppler[:-1, :] * np.conj(centered_range_doppler[1:, :])
                    instantaneous_freq_error = np.angle(phase_difference)
                    
                    # Average the error and compute the phase correction
                    avg_freq_error = np.mean(instantaneous_freq_error, axis=1)
                    phase_error = np.cumsum(np.insert(avg_freq_error, 0, 0))
                    
                    # Ensure phase correction has correct length
                    if len(phase_error) == chunk_rows:
                        phase_correction = np.exp(-1j * phase_error)[:, np.newaxis]
                    else:
                        # Pad or truncate if needed
                        phase_correction = np.ones((chunk_rows, 1), dtype=np.complex64)
                    
                    # Apply the correction
                    corrected_range_doppler = range_doppler_chunk * phase_correction
                    
                    # Transform back to image domain for the next iteration
                    data_chunk = ifft(ifftshift(corrected_range_doppler, axes=0), axis=0)
                else:
                    # --- MODIFICATION: Removed verbose warning ---
                    break
            
            # Place the focused chunk back into the main data array
            focused_data[:, start_col:end_col] = data_chunk
            # --- MODIFICATION: Removed per-chunk completion print statement ---
            
        except Exception as e:
            print(f"    Error processing chunk {start_col}-{end_col-1}: {e}")
            print(f"    Using original data for this chunk")
            focused_data[:, start_col:end_col] = complex_data[:, start_col:end_col]

    print("Autofocus complete.", flush=True)
    return focused_data