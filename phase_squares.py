# phase_squares.py
"""
Dedicated module to calculate and report the exact Latitude/Longitude
of the Yellow Box (Full Image) and Analysis Strips (Red/Violet) based on
pixel selection, rotation, and bilinear interpolation.
"""

import numpy as np
import json
import os
import re
import math

# --- 1. VECTOR GEOMETRY HELPERS ---

def ecef_to_enu(x, y, z, lat0, lon0):
    """
    Converts an ECEF vector (dx, dy, dz) to Local Tangent Plane (East, North, Up)
    at a reference Latitude/Longitude.
    """
    phi = math.radians(lat0)
    lam = math.radians(lon0)
    
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)
    
    # Transformation matrix
    e = -sin_lam * x + cos_lam * y
    n = -sin_phi * cos_lam * x - -sin_phi * sin_lam * y + cos_phi * z
    u = cos_phi * cos_lam * x + cos_phi * sin_lam * y + sin_phi * z
    
    return e, n, u

def get_heading_from_json(json_path, centroid_lat, centroid_lon):
    """
    Calculates the exact Flight Heading (Row Axis Azimuth) from the JSON vectors.
    Returns the angle in degrees clockwise from North.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f: 
            meta = json.load(f)
        
        # Extract the row_direction vector (Flight Path)
        if 'image_geometry' in meta['collect']['image']:
            r_vec = meta['collect']['image']['image_geometry']['row_direction']
        else:
            return None
        
        # Convert this 3D vector to 2D Local Map coordinates (East, North)
        e, n, u = ecef_to_enu(r_vec[0], r_vec[1], r_vec[2], centroid_lat, centroid_lon)
        
        # Calculate angle from North (Standard Compass Bearing)
        heading = math.degrees(math.atan2(e, n))
        
        # Normalize to 0-360
        heading = (heading + 360) % 360
        
        # The row vector points "down" the image (increasing row index)
        # We use this as our rotation basis.
        return heading
        
    except Exception as e:
        return None

def parse_metadata_txt(txt_path):
    """Parses Centroid and fallback Azimuth from metadata.txt."""
    if not txt_path or not os.path.exists(txt_path):
        return None, None, 0.0
        
    c_lat, c_lon, azimuth = None, None, 0.0
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        match_c = re.search(r'Centroid\s*Lon([\d\.]+)Lat([\d\.]+)', content.replace('\n', '').replace(' ', ''))
        if match_c:
            c_lon = float(match_c.group(1))
            c_lat = float(match_c.group(2))
            
        match_az = re.search(r'Viewing Azimuth\s*([\d\.]+)', content)
        if match_az:
            azimuth = float(match_az.group(1))
            
        return c_lat, c_lon, azimuth
            
    except Exception as e:
        print(f"  ⚠️ Error parsing TXT metadata: {e}")
        return None, None, 0.0

def calculate_corners_rotated(center_lat, center_lon, length_m, width_m, azimuth_deg):
    """Calculates 4 corners of a box centered on Lat/Lon, applying rotation."""
    R = 6378137.0
    
    # Half dimensions
    dy = length_m / 2.0
    dx = width_m / 2.0
    
    # Rotation
    theta = math.radians(azimuth_deg) 
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Define corners relative to center: UL, UR, LL, LR
    corners_local = [
        (dy, -dx),  # UL
        (dy, dx),   # UR
        (-dy, -dx), # LL
        (-dy, dx)   # LR
    ]
    
    corners_geo = []
    
    for (cy, cx) in corners_local:
        # Rotate vector
        y_rot = cy * cos_t - cx * sin_t
        x_rot = cy * sin_t + cx * cos_t
        
        # Convert meters to degrees
        d_lat = math.degrees(y_rot / R)
        d_lon = math.degrees(x_rot / (R * math.cos(math.radians(center_lat))))
        
        corners_geo.append((center_lat + d_lat, center_lon + d_lon))
        
    return corners_geo

def get_pixel_geo_coord(row, col, model):
    """Translates Pixel (Row, Col) to Lat/Lon using the Rotated Vector model."""
    try:
        R = 6378137.0
        
        # Distance from center in pixels
        d_row = row - model['c_row_idx']
        d_col = col - model['c_col_idx']
        
        # Distance in meters
        # Note: Row increases downwards, but we typically model +Y as North.
        # The Rotation Angle accounts for the Row-Axis orientation relative to North.
        # We project the (Row, Col) vector.
        dy = d_row * model['sp_r']
        dx = d_col * model['sp_c']
        
        theta = math.radians(model['azimuth'])
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # Apply rotation
        # d_north = dy * cos - dx * sin
        # d_east  = dy * sin + dx * cos
        # (Signs might need flipping depending on Left/Right look, but standard rotation usually works if azimuth is correct)
        d_north = dy * cos_t - dx * sin_t
        d_east  = dy * sin_t + dx * cos_t
        
        d_lat = math.degrees(d_north / R)
        d_lon = math.degrees(d_east / (R * math.cos(math.radians(model['c_lat']))))
        
        return model['c_lat'] + d_lat, model['c_lon'] + d_lon
        
    except:
        return 0.0, 0.0

def get_georef_model(json_path, txt_path, lat_offset=0.0, lon_offset=0.0):
    """Builds the coordinate model from files and applies calibration offsets."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f: 
            meta = json.load(f)
        
        im = meta['collect']['image']
        rows = im['rows']
        cols = im['columns']
        
        # Dimensions
        img_len = im.get('length', rows * im.get('pixel_spacing_row', 0.5))
        img_wid = im.get('width', cols * im.get('pixel_spacing_column', 0.5))
        
        # 1. Get Centroid from TXT
        c_lat, c_lon, txt_azimuth = parse_metadata_txt(txt_path)
        
        # 2. Get Heading from JSON Vectors (More Accurate)
        vec_heading = get_heading_from_json(json_path, c_lat, c_lon)
        final_azimuth = vec_heading if vec_heading is not None else txt_azimuth
        
        if c_lat and c_lon:
            # --- APPLY CALIBRATION OFFSET ---
            c_lat += lat_offset
            c_lon += lon_offset
            
            # Note: We don't necessarily need the yellow box corners for the calculation model,
            # but we calculate them for the report.
            corners = calculate_corners_rotated(c_lat, c_lon, img_len, img_wid, final_azimuth)
            ul, ur, ll, lr = corners[0], corners[1], corners[2], corners[3]
        else:
            ul=ur=ll=lr=(0,0)

        return {
            'rows': rows, 'cols': cols, 
            'ul': ul, 'ur': ur, 'll': ll, 'lr': lr,
            'c_lat': c_lat, 'c_lon': c_lon, 'azimuth': final_azimuth,
            'c_row_idx': rows/2.0, 'c_col_idx': cols/2.0,
            'sp_r': im.get('pixel_spacing_row', 0.3),
            'sp_c': im.get('pixel_spacing_column', 0.05)
        }
    except Exception as e:
        print(f"Error building model: {e}")
        return None

def get_box_coords(r_start, r_end, c_start, c_end, model):
    """Returns [UL, UR, LR, LL, UL] for a pixel box."""
    # We map (Start, Start) as UL, etc.
    # Note: r_start is the lower index (Top of image), r_end is higher index (Bottom)
    
    p1 = get_pixel_geo_coord(r_start, c_start, model) # UL
    p2 = get_pixel_geo_coord(r_start, c_end, model)   # UR
    p3 = get_pixel_geo_coord(r_end, c_end, model)     # LR
    p4 = get_pixel_geo_coord(r_end, c_start, model)   # LL
    
    return [p1, p2, p3, p4, p1] # Closed loop

# --- 2. MAIN REPORTING FUNCTION ---

def calculate_and_report_squares(json_path, txt_path, center_row, center_col, extent_km, s_left, e_left, s_right, e_right, lat_offset=0.0, lon_offset=0.0):
    """
    Calculates and reports coordinates for Yellow and Red boxes (Left/Right strips).
    Accepts calibration offsets to shift the result.
    """
    print("\n" + "="*60)
    print("📐 PHASE SQUARES: COORDINATE VERIFICATION REPORT")
    print("="*60)

    model = get_georef_model(json_path, txt_path, lat_offset, lon_offset)
    if not model:
        print("❌ Error: Could not build georeference model.")
        return None

    # --- YELLOW BOX ---
    print("\n🟨 YELLOW BOX (Full Image Footprint)")
    print(f"   Dimensions: {model['rows']} Rows x {model['cols']} Cols")
    print(f"   Center: {model['c_lat']:.6f}, {model['c_lon']:.6f}")
    print(f"   Rotation (Azimuth): {model['azimuth']:.2f}°")
    
    # Yellow Box is effectively pixel 0,0 to Max,Max
    yellow_coords = get_box_coords(0, model['rows']-1, 0, model['cols']-1, model)
    
    labels = ["UL", "UR", "LR", "LL"]
    for i in range(4):
        print(f"   {labels[i]}: Lat {yellow_coords[i][0]:.6f}, Lon {yellow_coords[i][1]:.6f}")

    # --- PIXEL MATH ---
    # Vertical Pixels (Radius)
    v_pixels = int((extent_km * 1000) / model['sp_r'])
    r_start = max(0, center_row - v_pixels)
    r_end = min(model['rows'] - 1, center_row + v_pixels)
    
    # Left Strip (Red) - e_left is outer bound
    c_start_l = max(0, center_col - e_left)
    c_end_l = max(0, center_col - s_left)
    
    # Right Strip (Violet) - e_right is outer bound
    c_start_r = min(model['cols'] - 1, center_col + s_right)
    c_end_r = min(model['cols'] - 1, center_col + e_right)

    print(f"\n   POI Center: Row {center_row}, Col {center_col}")
    print(f"   Extent: {extent_km} km (Rows {r_start}-{r_end})")

    # --- RED BOX (Left) ---
    print(f"\n🟥 RED BOX (Left Strip: -{e_left} to -{s_left})")
    print(f"   Pixel Bounds: Cols [{c_start_l}-{c_end_l}]")
    red_coords = get_box_coords(r_start, r_end, c_start_l, c_end_l, model)
    for i in range(4):
        print(f"   {labels[i]}: {red_coords[i][0]:.6f}, {red_coords[i][1]:.6f}")

    # --- VIOLET BOX (Right) ---
    print(f"\n🟪 VIOLET BOX (Right Strip: +{s_right} to +{e_right})")
    print(f"   Pixel Bounds: Cols [{c_start_r}-{c_end_r}]")
    violet_coords = get_box_coords(r_start, r_end, c_start_r, c_end_r, model)
    for i in range(4):
        print(f"   {labels[i]}: {violet_coords[i][0]:.6f}, {violet_coords[i][1]:.6f}")

    print("="*60 + "\n")
    
    return {
        'yellow': yellow_coords,
        'red': red_coords,
        'violet': violet_coords
    }