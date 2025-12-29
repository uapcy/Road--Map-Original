# phase_4_validation.py

import numpy as np

def geocode_tomographic_line_coords(radar_params, num_pixels):
    """
    Generates arrays of latitude and longitude coordinates for the tomographic line.
    This provides an accurate geographic reference for the validation plot.
    CORRECTED: This version interpolates both latitude and longitude for accuracy.

    Args:
        radar_params (dict): The dictionary of parsed parameters.
        num_pixels (int): The number of pixels in the tomographic line.

    Returns:
        (numpy.ndarray, numpy.ndarray): A tuple containing the latitude vector
                                        and the longitude vector. Returns (None, None)
                                        if coordinates are not found.
    """
    print("\n--- Starting Phase 4: Georeferencing ---")
    
    # Linearly interpolate between the corner coordinates from the radar parameters.
    lat_start = radar_params.get('lat_upper_left')
    lat_end = radar_params.get('lat_lower_left')
    lon_start = radar_params.get('lon_upper_left')
    lon_end = radar_params.get('lon_lower_left')
    
    if any(coord is None for coord in [lat_start, lat_end, lon_start, lon_end]):
        print("Could not find all corner coordinate parameters for georeferencing.")
        return None, None
        
    latitude_vector = np.linspace(lat_start, lat_end, num_pixels)
    longitude_vector = np.linspace(lon_start, lon_end, num_pixels)
    
    print(f"Generated latitude vector from {lat_start:.4f} to {lat_end:.4f}")
    print(f"Generated longitude vector from {lon_start:.4f} to {lon_end:.4f}")
    print("--- Georeferencing Complete ---")
    return latitude_vector, longitude_vector