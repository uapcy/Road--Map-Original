# phase_2_svd.py
"""
Phase 2: SVD Filter for Stripe Removal
Removes dominant vertical stripes (static clutter) from the micro-motion matrix
using Singular Value Decomposition (SVD).
"""

import numpy as np

def apply_svd_filter(Y_matrix, n_components=1):
    """
    Applies SVD filtering to remove the most dominant components (usually stripes/noise).
    
    Args:
        Y_matrix (np.ndarray): Complex displacement matrix (Pixels x Looks).
        n_components (int): Number of dominant singular values to remove (default 1).
        
    Returns:
        np.ndarray: Filtered Y_matrix with dominant components subtracted.
    """
    print(f"\n--- Applying Phase 2: SVD Filter (Removing {n_components} dominant components) ---", flush=True)
    
    # Check input dimensions
    if Y_matrix.shape[1] < n_components + 1:
        print("    Warning: Not enough looks for SVD filtering. Skipping.", flush=True)
        return Y_matrix

    try:
        # Perform Singular Value Decomposition
        # full_matrices=False ensures we get shapes (M, K), (K,), (K, N)
        U, S, Vt = np.linalg.svd(Y_matrix, full_matrices=False)
        
        # Create a copy of the Singular Values to modify
        S_clean = S.copy()
        
        # Zero out the top 'n' components (the static stripes)
        # These correspond to the largest singular values
        S_clean[:n_components] = 0.0
        
        # Reconstruct the matrix: U * diag(S_clean) * Vt
        # We use broadcasting for matrix multiplication with diagonal S
        Y_filtered = U @ np.diag(S_clean) @ Vt
        
        print("    SVD filtering complete.", flush=True)
        return Y_filtered

    except np.linalg.LinAlgError:
        print("    Error: SVD convergence failed. Returning original data.", flush=True)
        return Y_matrix
    except Exception as e:
        print(f"    Error in SVD filter: {e}", flush=True)
        return Y_matrix