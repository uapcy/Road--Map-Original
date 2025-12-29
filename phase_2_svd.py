# phase_2_svd.py
import numpy as np

def apply_svd_filter(Y_matrix, n_components=1):
    """
    Removes dominant static clutter (vertical stripes) using SVD.
    Essential for isolating micro-motions from terrain backscatter.
    """
    print(f"\n--- Applying Phase 2: SVD Clutter Suppression ---", flush=True)
    
    if Y_matrix.shape[1] < n_components + 1:
        return Y_matrix

    try:
        U, S, Vt = np.linalg.svd(Y_matrix, full_matrices=False)
        S_clean = S.copy()
        
        # Remove top components (static features)
        S_clean[:n_components] = 0.0
        
        Y_filtered = U @ np.diag(S_clean) @ Vt
        print("    SVD filtering complete.", flush=True)
        return Y_filtered

    except Exception as e:
        print(f"    SVD Error: {e}. Returning original.", flush=True)
        return Y_matrix