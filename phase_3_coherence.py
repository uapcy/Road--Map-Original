# phase_3_coherence.py

import numpy as np

def calculate_coherence_map(Y_matrix):
    """
    Calculates the infra-chromatic coherence for each pixel row in the Y matrix.

    Coherence is defined as the magnitude of the complex sum divided by the
    sum of the magnitudes. It is a measure of the phase stability of the signal.
    A value of 1.0 indicates a perfectly stable phase, while 0.0 indicates
    a completely random phase.

    Args:
        Y_matrix (np.ndarray): The complex-valued displacement matrix, with
                               shape (num_pixels, num_looks).

    Returns:
        np.ndarray: A 1D array of coherence values, with shape (num_pixels,).
    """
    print("\n--- Calculating Infra-Chromatic Coherence Map ---", flush=True)

    # Calculate the magnitude of the complex sum of the vectors (numerator)
    # This is |y1 + y2 + ... + yn|
    numerator = np.abs(np.sum(Y_matrix, axis=1))

    # Calculate the sum of the magnitudes of the vectors (denominator)
    # This is |y1| + |y2| + ... + |yn|
    denominator = np.sum(np.abs(Y_matrix), axis=1)

    # Initialize coherence map
    num_pixels = Y_matrix.shape[0]
    coherence_map = np.zeros(num_pixels, dtype=np.float32)

    # Avoid division by zero for pixels with no signal
    valid_indices = denominator > 1e-9
    coherence_map[valid_indices] = numerator[valid_indices] / denominator[valid_indices]

    print("--- Coherence Map Calculation Complete ---", flush=True)
    return coherence_map