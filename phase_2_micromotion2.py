# phase_2_micromotion2.py
# =============================================================================
#   MICRO-MOTION ESTIMATOR (V7-15 CHECKPOINT EDITION)
#   Updates: Added intra-column checkpointing to survive crashes during long loops.
# =============================================================================

import numpy as np
import os
from skimage.registration import phase_cross_correlation

def estimate_micro_motions_sliding_master(low_res_slcs, window_size=64, upsample_factor=1200, checkpoint_path=None):
    """
    Estimates micro-motions using 'Sliding Master' technique.
    
    New Feature:
    - checkpoint_path: If provided, saves progress here every 2000 steps (~10 mins).
      If file exists on startup, loads it and resumes loop.
    """
    print(f"\n--- Starting Phase 2.3: Seismic Micro-motion Estimation ---", flush=True)
    print(f"    Precision Mode: {upsample_factor}x upsampling", flush=True)
    
    num_looks = len(low_res_slcs)
    if num_looks < 2:
        return np.array([])

    num_pixels = len(low_res_slcs[0])
    half_win = window_size // 2
    
    # Differential matrix (velocity)
    # Initialize normally
    Y_diff = np.zeros((num_pixels, num_looks - 1), dtype=np.complex64)
    start_index = 0

    # --- CHECKPOINT LOADING ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            data = np.load(checkpoint_path)
            Y_diff_saved = data['Y_diff']
            saved_idx = int(data['last_index'])
            
            # Verify shape match
            if Y_diff_saved.shape == Y_diff.shape:
                Y_diff = Y_diff_saved
                start_index = saved_idx
                print(f"    [RESUME] Found checkpoint. Resuming at pixel {start_index}/{num_pixels}", flush=True)
            else:
                print(f"    [WARNING] Checkpoint shape mismatch. Starting over.", flush=True)
        except Exception as e:
            print(f"    [WARNING] Could not load checkpoint: {e}", flush=True)

    # --- MAIN LOOP ---
    CHECKPOINT_INTERVAL = 2000 # Save every ~2000 pixels (~10 mins at 150x)

    for i in range(start_index, num_pixels):
        # Progress Print
        if i % 500 == 0:
            print(f"    Progress: {i}/{num_pixels}", flush=True)
        
        # Checkpoint Save
        if checkpoint_path and i > 0 and i % CHECKPOINT_INTERVAL == 0:
            try:
                np.savez(checkpoint_path, Y_diff=Y_diff, last_index=i)
                print(f"    [SAVE] Checkpoint saved at {i}/{num_pixels}", flush=True)
            except:
                pass

        for j in range(num_looks - 1):
            try:
                master = low_res_slcs[j]
                slave = low_res_slcs[j+1]

                r_start = max(0, i - half_win)
                r_end = min(num_pixels, i + half_win)
                
                win_m = master[r_start:r_end]
                win_s = slave[r_start:r_end]

                if len(win_m) < 4 or np.mean(np.abs(win_m)) < 1e-6: 
                    Y_diff[i, j] = 1.0 + 0j
                    continue

                shift, error, phasediff = phase_cross_correlation(
                    win_m.reshape(-1, 1),
                    win_s.reshape(-1, 1),
                    upsample_factor=upsample_factor,
                    normalization=None
                )
                
                # Construct vibrational phasor from phase difference
                quality = max(0, 1.0 - error)
                Y_diff[i, j] = quality * np.exp(1j * phasediff)
                
            except:
                Y_diff[i, j] = 1.0 + 0j

    print("    Integrating differential shifts...", flush=True)
    
    # Clean up checkpoint after successful completion
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except: pass

    # Integration Step: Cumulative Product of Phasors
    # Z_t = Z_{t-1} * exp(j * dPhi)
    Y_integrated = np.cumprod(Y_diff, axis=1)
    
    # Pad with initial state
    Y_final = np.hstack((np.ones((num_pixels, 1), dtype=np.complex64), Y_integrated))
    
    return Y_final