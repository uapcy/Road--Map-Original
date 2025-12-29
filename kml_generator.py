# kml_generator.py

import os

def create_dumb_kml(output_filename, polygons_dict):
    """
    Writes KML file using pre-calculated coordinates passed from the main processor.
    NO MATH IS PERFORMED HERE to ensure exact match with visual selection.
    
    polygons_dict expected format:
    {
        'yellow': [(lat, lon), (lat, lon), (lat, lon), (lat, lon)],
        'red':    [(lat, lon)...],
        'violet': [(lat, lon)...]
    }
    """
    print(f"--- 📄 Writing KML to {output_filename} ---")
    
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>"""
    
    # Styles: Yellow (Outline), Red (Thick), Violet (Thick)
    styles = """
    <Style id="yellowStyle">
        <LineStyle><color>ff00ffff</color><width>2</width></LineStyle>
        <PolyStyle><fill>0</fill></PolyStyle>
    </Style>
    <Style id="redStyle">
        <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>
        <PolyStyle><fill>0</fill></PolyStyle>
    </Style>
    <Style id="violetStyle">
        <LineStyle><color>ffff00aa</color><width>3</width></LineStyle>
        <PolyStyle><fill>0</fill></PolyStyle>
    </Style>
"""
    
    body = ""
    
    # Define mapping for KML display names
    config = {
        'yellow': {'name': 'Full Analysis Extent', 'style': '#yellowStyle'},
        'red':    {'name': 'Left Strip', 'style': '#redStyle'},
        'violet': {'name': 'Right Strip', 'style': '#violetStyle'}
    }
    
    for key, coords in polygons_dict.items():
        if not coords: continue
        
        # KML format is "Lon,Lat,0" (Longitude first!)
        coord_str = " ".join([f"{lon},{lat},0" for lat, lon in coords])
        
        # Close the loop if not closed
        if coords[0] != coords[-1]:
            coord_str += f" {coords[0][1]},{coords[0][0]},0"
        
        name = config[key]['name']
        style = config[key]['style']
        
        body += f"""
    <Placemark>
        <name>{name}</name>
        <styleUrl>{style}</styleUrl>
        <Polygon>
            <tessellate>1</tessellate>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>{coord_str}</coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>"""

    kml_footer = "\n</Document>\n</kml>"

    try:
        with open(output_filename, 'w') as f:
            f.write(kml_header + styles + body + kml_footer)
        print(f"  ✓ Success. KML saved.")
    except Exception as e:
        print(f"  ❌ Error writing KML: {e}")