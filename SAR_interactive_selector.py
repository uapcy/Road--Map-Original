# SAR_interactive_selector.py

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re
import math
from skimage.transform import resize
import rasterio
import time
import datetime # Added for timestamping save file

# Import tkinter for the confirmation dialog
import tkinter as tk
from tkinter import messagebox

# --- HELPER: ECEF TO LAT/LON CONVERSION (WGS84) ---
def ecef_to_latlon(x, y, z):
    """
    Converts Earth-Centered, Earth-Fixed (ECEF) coordinates to Geodetic (Lat, Lon).
    Uses WGS84 ellipsoid constants.
    """
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 2*f - f*f
    ep2 = (a*a - b*b) / (b*b)
    
    p = math.sqrt(x**2 + y**2)
    th = math.atan2(a * z, b * p)
    
    lon = math.atan2(y, x)
    lat = math.atan2(z + ep2 * b * math.pow(math.sin(th), 3), 
                     p - e2 * a * math.pow(math.cos(th), 3))
    
    return math.degrees(lat), math.degrees(lon)

# --- ACCURATE COORDINATE CONVERSION USING PROPER BILINEAR INTERPOLATION ---
def get_pixel_geo_coord(row, col, model):
    """Convert pixel coordinates to geographic using bilinear interpolation between corners"""
    try:
        rows, cols = model['rows'], model['cols']
        
        # Get the known corner coordinates
        ul_lat, ul_lon = model['ul']
        ur_lat, ur_lon = model['ur']
        ll_lat, ll_lon = model['ll']
        lr_lat, lr_lon = model['lr']
        
        # Normalize pixel coordinates to [0,1] range
        row_norm = row / (rows - 1) if rows > 1 else 0
        col_norm = col / (cols - 1) if cols > 1 else 0
        
        # Clamp to [0,1] range
        row_norm = max(0.0, min(1.0, row_norm))
        col_norm = max(0.0, min(1.0, col_norm))
        
        # Bilinear interpolation between the four corners
        # Top edge interpolation
        top_lat = ul_lat + (ur_lat - ul_lat) * col_norm
        top_lon = ul_lon + (ur_lon - ul_lon) * col_norm
        
        # Bottom edge interpolation
        bottom_lat = ll_lat + (lr_lat - ll_lat) * col_norm
        bottom_lon = ll_lon + (lr_lon - ll_lon) * col_norm
        
        # Vertical interpolation between top and bottom
        lat = top_lat + (bottom_lat - top_lat) * row_norm
        lon = top_lon + (bottom_lon - top_lon) * row_norm
        
        return lat, lon
        
    except Exception as e:
        print(f"Error in get_pixel_geo_coord: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def geo_to_pixel(lat, lon, model):
    """Convert geographic coordinates to pixel using proper inverse bilinear interpolation"""
    try:
        rows, cols = model['rows'], model['cols']
        
        # Get the known corner coordinates
        ul_lat, ul_lon = model['ul']
        ur_lat, ur_lon = model['ur']
        ll_lat, ll_lon = model['ll']
        lr_lat, lr_lon = model['lr']
        
        # Simple bounds check
        min_lat = min(ul_lat, ur_lat, ll_lat, lr_lat)
        max_lat = max(ul_lat, ur_lat, ll_lat, lr_lat)
        min_lon = min(ul_lon, ur_lon, ll_lon, lr_lon)
        max_lon = max(ul_lon, ur_lon, ll_lon, lr_lon)
        
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return None, None
        
        # Use iterative optimization to find the best pixel coordinates
        # This solves the inverse of the bilinear interpolation problem
        
        def objective_function(params):
            row_norm, col_norm = params
            # Clamp to [0,1] range
            row_norm = max(0.0, min(1.0, row_norm))
            col_norm = max(0.0, min(1.0, col_norm))
            
            # Forward transformation (same as get_pixel_geo_coord)
            top_lat = ul_lat + (ur_lat - ul_lat) * col_norm
            top_lon = ul_lon + (ur_lon - ul_lon) * col_norm
            bottom_lat = ll_lat + (lr_lat - ll_lat) * col_norm
            bottom_lon = ll_lon + (lr_lon - ll_lon) * col_norm
            pred_lat = top_lat + (bottom_lat - top_lat) * row_norm
            pred_lon = top_lon + (bottom_lon - top_lon) * row_norm
            
            # Calculate error
            lat_error = pred_lat - lat
            lon_error = pred_lon - lon
            return lat_error**2 + lon_error**2
        
        # Initial guess - center of image
        initial_guess = [0.5, 0.5]
        
        # Use scipy optimization if available, otherwise use simple grid search
        try:
            from scipy.optimize import minimize
            result = minimize(objective_function, initial_guess, method='L-BFGS-B', 
                            bounds=[(0, 1), (0, 1)], options={'ftol': 1e-10})
            if result.success:
                row_norm, col_norm = result.x
            else:
                raise ValueError("Optimization failed")
        except (ImportError, ValueError):
            # Fallback: grid search
            best_error = float('inf')
            best_row_norm, best_col_norm = 0.5, 0.5
            
            # Sample the parameter space
            for i in range(21):  # 0.0 to 1.0 in steps of 0.05
                for j in range(21):
                    row_norm_test = i * 0.05
                    col_norm_test = j * 0.05
                    error = objective_function([row_norm_test, col_norm_test])
                    if error < best_error:
                        best_error = error
                        best_row_norm, best_col_norm = row_norm_test, col_norm_test
            
            row_norm, col_norm = best_row_norm, best_col_norm
        
        # Convert normalized coordinates to pixel coordinates
        row = int(round(row_norm * (rows - 1)))
        col = int(round(col_norm * (cols - 1)))
        
        # Clamp to image bounds
        row = max(0, min(row, rows - 1))
        col = max(0, min(col, cols - 1))
        
        return row, col
        
    except Exception as e:
        print(f"Error in geo_to_pixel: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def debug_coordinate_system(model):
    """Comprehensive debugging of the coordinate system - SIMPLIFIED AND ACCURATE"""
    print("\n" + "="*80)
    print("COORDINATE SYSTEM DEBUG ANALYSIS")
    print("="*80)
    
    # 1. Basic image info
    print("\n1. IMAGE DIMENSIONS:")
    print(f"   Rows: {model['rows']}, Columns: {model['cols']}")
    
    # 2. Corner coordinates from JSON (known to be correct)
    print("\n2. KNOWN CORNER COORDINATES (from JSON):")
    print(f"   Upper-Left:  Lat {model['ul'][0]:.6f}, Lon {model['ul'][1]:.6f}")
    print(f"   Upper-Right: Lat {model['ur'][0]:.6f}, Lon {model['ur'][1]:.6f}") 
    print(f"   Lower-Left:  Lat {model['ll'][0]:.6f}, Lon {model['ll'][1]:.6f}")
    print(f"   Lower-Right: Lat {model['lr'][0]:.6f}, Lon {model['lr'][1]:.6f}")
    
    # 3. Test corner conversions
    print("\n3. CORNER COORDINATE VERIFICATION:")
    corners = {
        'Upper-Left': (0, 0),
        'Upper-Right': (0, model['cols']-1),
        'Lower-Left': (model['rows']-1, 0),
        'Lower-Right': (model['rows']-1, model['cols']-1)
    }
    
    corner_mapping = {
        'Upper-Left': 'ul',
        'Upper-Right': 'ur', 
        'Lower-Left': 'll',
        'Lower-Right': 'lr'
    }
    
    for corner_name, (row, col) in corners.items():
        lat, lon = get_pixel_geo_coord(row, col, model)
        if lat is not None:
            # Compare with known corner coordinates
            corner_key = corner_mapping[corner_name]
            expected_lat, expected_lon = model[corner_key]
            lat_diff = abs(lat - expected_lat)
            lon_diff = abs(lon - expected_lon)
            status = "✓" if lat_diff < 0.0001 and lon_diff < 0.0001 else "✗"
            print(f"   {status} {corner_name:12} - Pixel: ({row:6d}, {col:5d})")
            print(f"        Expected: Lat {expected_lat:.6f}, Lon {expected_lon:.6f}")
            print(f"        Got:      Lat {lat:.6f}, Lon {lon:.6f}")
            print(f"        Diff:     Lat {lat_diff:.6f}, Lon {lon_diff:.6f}")
    
    # 4. Test coordinate consistency - THE KEY TEST
    print("\n4. COORDINATE CONSISTENCY TEST (Round-trip accuracy):")
    test_points = [
        ('Center', model['rows']//2, model['cols']//2),
        ('Upper-Left', 0, 0),
        ('Upper-Right', 0, model['cols']-1),
        ('Lower-Left', model['rows']-1, 0),
        ('Lower-Right', model['rows']-1, model['cols']-1),
        ('Quarter', model['rows']//4, model['cols']//4),
        ('Three-Quarter', 3*model['rows']//4, 3*model['cols']//4),
    ]
    
    max_row_error = 0
    max_col_error = 0
    successful_tests = 0
    
    for point_name, row, col in test_points:
        if 0 <= row < model['rows'] and 0 <= col < model['cols']:
            # Pixel → Geo → Pixel round trip
            lat, lon = get_pixel_geo_coord(row, col, model)
            if lat is not None:
                back_row, back_col = geo_to_pixel(lat, lon, model)
                if back_row is not None and back_col is not None:
                    row_diff = abs(back_row - row)
                    col_diff = abs(back_col - col)
                    max_row_error = max(max_row_error, row_diff)
                    max_col_error = max(max_col_error, col_diff)
                    
                    # For a 107k x 11k image, we can tolerate small errors due to interpolation
                    status = "✓" if row_diff <= 1 and col_diff <= 1 else "✗"
                    if status == "✓":
                        successful_tests += 1
                    
                    print(f"   {status} {point_name:18} - Original: ({row:5d}, {col:5d}) -> Back: ({back_row:5d}, {back_col:5d}) | Diff: ({row_diff:2d}, {col_diff:2d})")
    
    print(f"\n   Maximum round-trip error: ({max_row_error} rows, {max_col_error} columns)")
    print(f"   Successful tests: {successful_tests}/{len(test_points)}")
    
    if max_row_error <= 1 and max_col_error <= 1:
        print("   ✓ Coordinate system is ACCURATE ENOUGH for practical use!")
    else:
        print("   ✗ Coordinate system needs improvement")
    
    print("\n" + "="*80)

def get_accurate_georef_model(json_path, txt_path):
    """Build georeferencing model using JSON metadata with Fallback Calculation"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f: 
            meta = json.load(f)
        
        collect = meta.get('collect', {})
        image_meta = collect.get('image', {})
        geom = image_meta.get('image_geometry', {})
        
        # Get image dimensions
        rows = image_meta.get('rows', 10000)
        cols = image_meta.get('columns', 10000)
        
        # --- ATTEMPT 1: EXPLICIT GEO KEYS (Common in GEO products) ---
        try:
            lat_ul = geom['latitude_upper_left']
            lon_ul = geom['longitude_upper_left']
            lat_ur = geom['latitude_upper_right']
            lon_ur = geom['longitude_upper_right']
            lat_ll = geom['latitude_lower_left']
            lon_ll = geom['longitude_lower_left']
            lat_lr = geom['latitude_lower_right']
            lon_lr = geom['longitude_lower_right']
            
            ul = (lat_ul, lon_ul)
            ur = (lat_ur, lon_ur)
            ll = (lat_ll, lon_ll)
            lr = (lat_lr, lon_lr)
            
            print("   [GEOREF] Found explicit corner coordinates in JSON.")

        except KeyError:
            # --- ATTEMPT 2: CALCULATION FROM ECEF CENTER (SLC/Spotlight products) ---
            print("   [GEOREF] Explicit corners missing. Calculating from ECEF Center (Spotlight Mode)...")
            
            # 1. Get ECEF Center
            # Try 'center_pixel' first
            center_pixel = image_meta.get('center_pixel', {})
            tgt_pos = center_pixel.get('target_position') # [X, Y, Z]
            
            if not tgt_pos:
                # Fallback to SRP if center pixel is missing
                tgt_pos = geom.get('scene_reference_point_ecef')
                
            if not tgt_pos:
                print("   [GEOREF CRITICAL] Could not find ECEF center. Defaulting to (0,0).")
                c_lat, c_lon = 0.0, 0.0
            else:
                c_lat, c_lon = ecef_to_latlon(tgt_pos[0], tgt_pos[1], tgt_pos[2])
                print(f"   [GEOREF] Calculated Center: Lat {c_lat:.4f}, Lon {c_lon:.4f}")

            # 2. Get Spacing
            # Use 'pixel_spacing_row/column' or 'row/col_sample_spacing'
            sp_r = image_meta.get('pixel_spacing_row') or geom.get('row_sample_spacing', 1.0)
            sp_c = image_meta.get('pixel_spacing_column') or geom.get('col_sample_spacing', 1.0)
            
            # 3. Calculate Extents (Meters)
            half_height_m = (rows * sp_r) / 2.0
            half_width_m = (cols * sp_c) / 2.0
            
            # 4. Convert to Degrees (Approximate for Visualization)
            # 1 deg lat ~= 111,111 m
            # 1 deg lon ~= 111,111 * cos(lat) m
            lat_deg_per_m = 1.0 / 111111.0
            lon_deg_per_m = 1.0 / (111111.0 * math.cos(math.radians(c_lat)))
            
            d_lat = half_height_m * lat_deg_per_m
            d_lon = half_width_m * lon_deg_per_m
            
            # 5. Define Synthetic Corners (North-Up)
            # Row index 0 is Top (North), Col index 0 is Left (West) usually
            ul = (c_lat + d_lat, c_lon - d_lon)
            ur = (c_lat + d_lat, c_lon + d_lon)
            ll = (c_lat - d_lat, c_lon - d_lon)
            lr = (c_lat - d_lat, c_lon + d_lon)

        model = {
            'rows': rows, 
            'cols': cols, 
            'ul': ul, 'ur': ur, 'll': ll, 'lr': lr,
            'image_geometry': geom
        }
        
        # Run comprehensive debugging
        debug_coordinate_system(model)
        
        return model
    except Exception as e: 
        print(f"FATAL: Could not build JSON georef model: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_column_range_input(prompt, default=None):
    while True:
        user_input = input(prompt).strip()
        if not user_input and default is not None: return default[0], default[1]
        range_match = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', user_input)
        if range_match:
            try:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if start > end: print("Invalid range: Start > End."); continue
                return start, end
            except ValueError: print("Invalid number format."); continue
        single_match = re.match(r'^\s*(\d+)\s*$', user_input)
        if single_match:
            try: return 0, int(single_match.group(1))
            except ValueError: print("Invalid number format."); continue
        if not user_input: return 0,0
        print("Invalid format. Use '30' or '30-50'.")

# --- GUI AND INTERACTION LOGIC ---
class PointDrawer:
    def __init__(self, ax, georef_model):
        self.ax = ax
        self.model = georef_model
        self.artists = []
        self.points = []
        self.point_artists = [] 
        # FIX: Store connection ID so we can disconnect later
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self)
        self.last_printed_point = None  # Track last printed point to avoid duplicates

    def __call__(self, event):
        if event.inaxes != self.ax or event.button != 1: return
        # When clicking, we get geographic coordinates from the map display
        # Convert these directly to pixel coordinates
        lat, lon = event.ydata, event.xdata
        row, col = geo_to_pixel(lat, lon, self.model)
        if row is None: return
        
        self.clear_last_point() 
        
        if self.points: self.points[-1] = (row, col)
        else: self.points.append((row, col))
        
        marker, = self.ax.plot(lon, lat, 'go', markersize=8, alpha=0.8)
        lat_offset = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.02 
        label_text = f'  Lon: {lon:.5f}, Lat: {lat:.5f}\n  [Row: {row}, Col: {col}]'
        label = self.ax.text(lon, lat + lat_offset, label_text, color='white', backgroundcolor=(0,0,0,0.6), ha='center', va='bottom', fontsize=12)
        
        self.point_artists.append(marker)
        self.point_artists.append(label)

        # Only print if this is a new point (not duplicate from mouse movement)
        current_point = (row, col)
        if current_point != self.last_printed_point:
            print(f"\n--- Point Selected ---\n  Pixel: (Row: {row}, Col: {col})\n  Geo: (Lon: {lon:.5f}, Lat: {lat:.5f})")
            self.last_printed_point = current_point
        
        self.ax.figure.canvas.draw_idle()
    
    def clear_last_point(self):
        for artist in self.point_artists:
            artist.remove()
        self.point_artists.clear()

    # FIX: New method to stop listening for clicks
    def disconnect(self):
        if self.cid is not None:
            self.ax.figure.canvas.mpl_disconnect(self.cid)
            self.cid = None
            # print("Interactive map disconnected.")

    def draw_point_at_geo(self, lat, lon):
        """Convert geographic coordinates to pixel using JSON georeferencing"""
        row, col = geo_to_pixel(lat, lon, self.model)
        if row is None: return
        
        self.clear_last_point() 
        
        if self.points: self.points[-1] = (row, col)
        else: self.points.append((row, col))
        
        marker, = self.ax.plot(lon, lat, 'go', markersize=8, alpha=0.8)
        lat_offset = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.02 
        label_text = f'  Lon: {lon:.5f}, Lat: {lat:.5f}\n  [Row: {row}, Col: {col}]'
        label = self.ax.text(lon, lat + lat_offset, label_text, color='white', backgroundcolor=(0,0,0,0.6), ha='center', va='bottom', fontsize=12)
        
        self.point_artists.append(marker)
        self.point_artists.append(label)

        print(f"\n--- Point Selected ---\n  Pixel: (Row: {row}, Col: {col})\n  Geo: (Lon: {lon:.5f}, Lat: {lat:.5f})")
        self.ax.figure.canvas.draw_idle()

    def set_point_from_pixel(self, row, col):
        """Convert pixel coordinates to geographic using JSON georeferencing"""
        lat, lon = get_pixel_geo_coord(row, col, self.model)
        if lat is None or lon is None:
            print(f"Error: Could not find geo-coordinate for pixel ({row}, {col}).")
            return

        # Clear any previous point
        self.clear_last_point() 
        
        # Store the original pixel coordinates directly
        if self.points:
            self.points[-1] = (row, col)
        else:
            self.points.append((row, col))
        
        # Draw the marker at the calculated geographic coordinates
        marker, = self.ax.plot(lon, lat, 'go', markersize=8, alpha=0.8)
        lat_offset = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.02 
        label_text = f'  Lon: {lon:.5f}, Lat: {lat:.5f}\n  [Row: {row}, Col: {col}]'
        label = self.ax.text(lon, lat + lat_offset, label_text, color='white', 
                            backgroundcolor=(0,0,0,0.6), ha='center', va='bottom', fontsize=12)
        
        self.point_artists.append(marker)
        self.point_artists.append(label)

        print(f"\n--- Point Selected ---\n  Pixel: (Row: {row}, Col: {col})\n  Geo: (Lon: {lon:.5f}, Lat: {lat:.5f})")
        self.ax.figure.canvas.draw_idle()

    def draw_analysis_area(self, bounds):
        # Check if window is still open before drawing to prevent errors
        if not plt.fignum_exists(self.ax.figure.number):
            return

        row_start, row_end = bounds['row_start'], bounds['row_end']
        col_start, col_end = bounds['col_start'], bounds['col_end']
        
        # Use JSON georeferencing for all corner calculations
        ul_lat, ul_lon = get_pixel_geo_coord(row_start, col_start, self.model)
        ur_lat, ur_lon = get_pixel_geo_coord(row_start, col_end, self.model)
        lr_lat, lr_lon = get_pixel_geo_coord(row_end, col_end, self.model)
        ll_lat, ll_lon = get_pixel_geo_coord(row_end, col_start, self.model)
        
        poly = plt.Polygon([(ul_lon, ul_lat), (ur_lon, ur_lat), (lr_lon, lr_lat), (ll_lon, ll_lat)],
                           closed=True, edgecolor='red', facecolor='none', linewidth=2)
        self.ax.add_patch(poly)
        self.artists.append(poly)
        self.ax.figure.canvas.draw_idle()

    def clear_all(self):
        for artist in self.artists: artist.remove()
        self.artists.clear()
        self.clear_last_point()

def setup_combined_axes_labels(ax, model):
    fontsize = 12
    yticks = ax.get_yticks()
    ylabels = []
    mid_lon = np.mean(ax.get_xlim())
    for lat in yticks:
        row, _ = geo_to_pixel(lat, mid_lon, model)
        if row is not None:
            ylabels.append(f"{lat:.2f}°N\n(Row: {row})")
        else:
            ylabels.append(f"{lat:.2f}°N")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=fontsize)

    xticks = ax.get_xticks()
    xlabels = []
    mid_lat = np.mean(ax.get_ylim())
    for lon in xticks:
        _, col = geo_to_pixel(mid_lat, lon, model)
        if col is not None:
            xlabels.append(f"{lon:.2f}°E\n(Col: {col})")
        else:
            xlabels.append(f"{lon:.2f}°E")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=fontsize)

def run_interactive_selector(file_paths, config, preloaded_data=None):
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        print("Warning: Could not initialize Tkinter. Confirmation dialogs will be skipped.")
        messagebox = None
    
    print("--- 🗺️ Starting Interactive Point Selector ---", flush=True)
    
    # Use JSON georeferencing (the correct approach)
    georef_model = get_accurate_georef_model(file_paths['json_file'], file_paths['txt_file'])
    if not georef_model: 
        print("ERROR: Could not build georeferencing model from JSON.")
        return None, None, None

    print("\n--- 🔎 SAR Image Details ---")
    print(f"  Dimensions: {georef_model['rows']} rows x {georef_model['cols']} columns")
    print(f"  Upper-Left Corner: Lat {georef_model['ul'][0]:.6f}, Lon {georef_model['ul'][1]:.6f}")
    print(f"  Upper-Right Corner: Lat {georef_model['ur'][0]:.6f}, Lon {georef_model['ur'][1]:.6f}")
    print(f"  Lower-Left Corner: Lat {georef_model['ll'][0]:.6f}, Lon {georef_model['ll'][1]:.6f}")
    print(f"  Lower-Right Corner: Lat {georef_model['lr'][0]:.6f}, Lon {georef_model['lr'][1]:.6f}")
    
    if 'image_geometry' in georef_model:
        geom = georef_model['image_geometry']
        print(f"  Column (Azimuth) Spacing: {geom.get('col_sample_spacing', 'N/A'):.4f} m/pixel")
        print(f"  Row (Range) Spacing:     {geom.get('row_sample_spacing', 'N/A'):.4f} m/pixel")
    print("---------------------------\n", flush=True)
    
    log_mag = None
    
    # --- CHECK FOR PRELOADED DATA ---
    if preloaded_data is not None:
        print("Using preloaded visualization data (Cache Hit).", flush=True)
        log_mag = preloaded_data
    else:
        try:
            print("Loading full-resolution SAR data from TIFF. This may take a moment for large files...", flush=True)
            start_time = time.time()
            with rasterio.open(file_paths['tiff_file']) as src:
                complex_data = src.read(1)
            print(f"  ✓ Data loaded in {time.time() - start_time:.2f} seconds.", flush=True)
            
            print("Processing data for visualization (downsampling, rotating, scaling)...", flush=True)
            start_time = time.time()
            
            magnitude = np.abs(complex_data)
            del complex_data
            
            longest_side = max(magnitude.shape)
            downsample_factor = max(1, longest_side // 1000) 
            small_mag = resize(magnitude, 
                            (magnitude.shape[0] // downsample_factor, magnitude.shape[1] // downsample_factor), 
                            anti_aliasing=True)
            small_mag = np.fliplr(small_mag)
            small_mag = np.rot90(small_mag, k=2)
            small_mag = np.rot90(small_mag, k=1)
            log_mag = np.log1p(small_mag)
            print(f"  ✓ Visualization data prepared in {time.time() - start_time:.2f} seconds.", flush=True)
        except Exception as e:
            print(f"Fatal: Could not load TIFF. Error: {e}")
            return None, None, None

    print("Generating and displaying the interactive map...", flush=True)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Use JSON bounds for geographic extent (Paris coordinates)
    min_lon = min(georef_model['ul'][1], georef_model['ll'][1], georef_model['ur'][1], georef_model['lr'][1])
    max_lon = max(georef_model['ul'][1], georef_model['ll'][1], georef_model['ur'][1], georef_model['lr'][1])
    min_lat = min(georef_model['ul'][0], georef_model['ll'][0], georef_model['ur'][0], georef_model['lr'][0])
    max_lat = max(georef_model['ul'][0], georef_model['ll'][0], georef_model['ur'][0], georef_model['lr'][0])

    extent_geo = [min_lon, max_lon, min_lat, max_lat]
    
    im = ax.imshow(log_mag, cmap='gray', aspect='auto', extent=extent_geo)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log Magnitude', fontsize=12, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title('Interactive Point Selector (Using CORNER-BASED Georeferencing)', fontsize=16)
    ax.set_xlabel('Longitude (°E)', fontsize=14)
    ax.set_ylabel('Latitude (°N)', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.5)

    # Tooltip using JSON georeferencing
    def format_coord(x, y):
        # x is Lon, y is Lat
        row, col = geo_to_pixel(y, x, georef_model)
        if row is not None and col is not None:
            return f'Lon: {x:.6f}°E, Lat: {y:.6f}°N | Pixel: Row {row}, Col {col}'
        else:
            return f'Lon: {x:.6f}°E, Lat: {y:.6f}°N | Outside image bounds'
    
    ax.format_coord = format_coord
    
    setup_combined_axes_labels(ax, georef_model)
    
    fig.tight_layout(pad=2.5)
    
    point_drawer = PointDrawer(ax, georef_model)
    # MODIFIED: block=False to allow code execution to continue
    plt.show(block=False)
    plt.pause(0.1)
    
    print("\n--- 🗺️ Map is active. Select points or type a command. ---", flush=True)
    print("NOTE: Using CORNER-BASED georeferencing for accurate Paris coordinates")
    print(f"Image covers area: Lon {min_lon:.4f}°E to {max_lon:.4f}°E, Lat {min_lat:.4f}°N to {max_lat:.4f}°N")
    
    selected_point = None
    while True:
        # MODIFIED: Check if window closed AND if we have a selection.
        # If window closed but we have a point, treat it as "finish".
        if not plt.fignum_exists(fig.number):
            if point_drawer.points:
                selected_point = point_drawer.points[-1]
                print("\nWindow closed by user. Proceeding with last selected point.")
                break
            else:
                print("Window closed before selection was finalized. Exiting.")
                return None, None, None
        
        # We still need a loop to get user input for mode selection or finalization,
        # but the map window remains interactive.
        try:
            choice = input("\nEnter mode: (c)lick info | (t)ype row/col | (s)et lon/lat | (f)inish | (q)uit: ").lower()
        except EOFError:
            # Handle case where input stream is broken (e.g. forced close)
            return None, None, None

        if choice == 'c':
            print("  > Click is always active. Simply click a point on the map.")
            print("  > Use 'f' to finalize your selection with the last point chosen.")
            continue
        
        elif choice == 't':
            try:
                r, c = map(int, input("  > Enter Row, Col (e.g., 64645, 7006): ").replace(" ", "").split(','))
                point_drawer.set_point_from_pixel(r, c)
            except (ValueError, IndexError):
                print("  > Invalid format.")
            continue

        elif choice == 's':
            try:
                lon, lat = map(float, input("  > Enter Lon, Lat (e.g., 2.2945, 48.8583): ").replace(" ", "").split(','))
                point_drawer.draw_point_at_geo(lat, lon)
            except (ValueError, IndexError):
                print("  > Invalid format.")
            continue
            
        elif choice == 'f':
            if not point_drawer.points:
                print("  > No point has been selected. Please select a point before finishing.")
                continue
            else:
                selected_point = point_drawer.points[-1]
                print(f"\n--- Final point chosen: (Row: {selected_point[0]}, Col: {selected_point[1]}) ---")
                break
        
        elif choice == 'q':
            # MODIFIED: Removing closing logic since we want to keep map open, 
            # but 'q' implies stopping the process.
            if messagebox is None or messagebox.askyesno("Confirm Quit", "Are you sure you want to quit the selection process?"):
                print("Selection cancelled by user.")
                # We do close here because the user asked to QUIT the entire process.
                if plt.fignum_exists(fig.number):
                    plt.close(fig) 
                return None, None, None
            else:
                print("Quit cancelled.")
                continue
        else:
            print("  > Invalid choice.")

    center_row, central_col_idx = selected_point
    
    # --- FIX: Calculate precise geographic coordinates for return ---
    # Use the robust method to get the exact lat/lon of the center point
    lat_center, lon_center = get_pixel_geo_coord(center_row, central_col_idx, georef_model)
    
    # FIX: Disconnect mouse event handler so clicks don't register after we're done
    point_drawer.disconnect()
    
    print("\n--- Define Analysis Area ---")
    
    def_ext = float(config.get('user_parameters', {}).get('ANALYSIS_EXTENT_KM', 0.7))
    ext_km = float(input(f"Enter ANALYSIS_EXTENT_KM (default: {def_ext}): ") or def_ext)
    
    def_rng = config.get('column_ranges', {'start_left':0, 'end_left':30, 'start_right':1, 'end_right':30})
    s_l, e_l = parse_column_range_input(f"Cols LEFT of center ({central_col_idx}) (def: {def_rng['start_left']}-{def_rng['end_left']}): ", (def_rng['start_left'], def_rng['end_left']))
    s_r, e_r = parse_column_range_input(f"Cols RIGHT of center ({central_col_idx}) (def: {def_rng['start_right']}-{def_rng['end_right']}): ", (def_rng['start_right'], def_rng['end_right']))
    
    # Use azimuth spacing from JSON for accurate pixel extent calculation
    if 'image_geometry' in georef_model:
        azimuth_spacing = georef_model['image_geometry']['col_sample_spacing']
    else:
        azimuth_spacing = 0.04667  # Default from your JSON data
    
    ext_pix = int(round((ext_km * 1000) / azimuth_spacing))
    r_start, r_end = max(0, center_row - ext_pix), min(georef_model['rows'] - 1, center_row + ext_pix)
    cols_set = set(range(central_col_idx - e_l, central_col_idx - s_l)) | {central_col_idx} | set(range(central_col_idx + s_r, central_col_idx + e_r + 1))
    valid_cols = [c for c in cols_set if 0 <= c < georef_model['cols']]
    
    if not valid_cols:
        print("Error: No valid columns selected for analysis.")
        if plt.fignum_exists(fig.number):
            plt.close(fig)
        return None, None, None

    bounds = {'row_start':r_start, 'row_end':r_end, 'col_start':min(valid_cols), 'col_end':max(valid_cols)}
    point_drawer.draw_analysis_area(bounds)
    ax.set_title('Final Selection and Analysis Area')
    
    if plt.fignum_exists(fig.number):
        fig.canvas.draw_idle()
    
    # --- ADDED: SAVE MAP PROMPT ---
    # Pause slightly to ensure the red box is drawn
    plt.pause(0.5)
    save_map = input("\nDo you want to save the selection map image? (y/n): ").lower().strip()
    if save_map in ['y', 'yes']:
        try:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            map_filename = f"selection_map_{ts}.png"
            fig.savefig(map_filename, dpi=150, bbox_inches='tight')
            print(f"  ✓ Selection map saved to {map_filename}")
        except Exception as e:
            print(f"  ⚠️ Could not save map: {e}")

    # --- ADDED: CLOSE WINDOW ---
    print("Closing interactive map window...")
    plt.close(fig)
    
    final_selection = {
        "center_row": center_row, "center_col": central_col_idx,
        "lat_center": lat_center, "lon_center": lon_center, 
        "analysis_extent_km": ext_km,
        "start_left": s_l, "end_left": e_l, "start_right": s_r, "end_right": e_r,
        # NEW: Pass valid corners back to main processor
        "corners": {
            'ul': georef_model['ul'],
            'ur': georef_model['ur'],
            'll': georef_model['ll'],
            'lr': georef_model['lr'],
            # --- FIX: Include rows/cols for KML generator compatibility ---
            'rows': georef_model['rows'],
            'cols': georef_model['cols']
        }
    }
    
    # --- CONSTRUCT NEW DEFAULTS FOR RETURN (Fixes the Unpacking Crash) ---
    new_defaults = {
        'calibration': {
            'center_row': center_row,
            'center_col': central_col_idx
        },
        'user_parameters': {
            'ANALYSIS_EXTENT_KM': ext_km
        },
        'column_ranges': {
            'start_left': s_l, 'end_left': e_l,
            'start_right': s_r, 'end_right': e_r
        }
    }
    
    print("\nAnalysis area confirmed.")
    
    # Return 3 items to match phase_image_handling.py expectations
    return final_selection, log_mag, new_defaults