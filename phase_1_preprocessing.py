# phase_1_preprocessing.py
import numpy as np
from scipy.fft import fft
def extract_tomographic_line(complex_data, column_index):
    print(f"\nExtracting tomographic line from column {column_index}...", flush=True)
    tomographic_line = complex_data[:, column_index]
    print(f"Extracted line with shape: {tomographic_line.shape}", flush=True)
    return tomographic_line
def transform_to_frequency_domain(tomographic_line):
    print("\nApplying Fast Fourier Transform (FFT)...", flush=True)
    tomographic_line_fft = fft(tomographic_line)
    print("FFT complete.", flush=True)
    return tomographic_line_fft