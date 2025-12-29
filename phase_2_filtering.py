# phase_2_filtering.py
import numpy as np

def _kalman_filter_1d(measurements, process_noise, measurement_noise):
    A = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = process_noise * np.eye(2)
    R = np.array([[measurement_noise]])
    x = np.array([measurements[0], 0])
    P = np.eye(2)
    
    filtered = np.zeros(len(measurements))
    filtered[0] = measurements[0]
    
    for i in range(1, len(measurements)):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        
        y = measurements[i] - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        x = x_pred + K @ y
        P = (np.eye(2) - K @ H) @ P_pred
        filtered[i] = x[0]
        
    return filtered

def apply_kalman_filter(Y_matrix, process_noise=0.01, measurement_noise=0.1):
    print(f"\n--- Applying Kalman Filter ---", flush=True)
    num_pixels, num_looks = Y_matrix.shape
    if num_looks < 2: return Y_matrix
    
    Y_filtered = np.zeros_like(Y_matrix)
    for i in range(num_pixels):
        real = _kalman_filter_1d(np.real(Y_matrix[i]), process_noise, measurement_noise)
        imag = _kalman_filter_1d(np.imag(Y_matrix[i]), process_noise, measurement_noise)
        Y_filtered[i] = real + 1j * imag
        
    return Y_filtered