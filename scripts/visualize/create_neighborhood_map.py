#!/usr/bin/env python3
"""
Create interactive choropleth map of NYC neighborhoods by Child Health Housing Index.

This script generates:
1. Main choropleth map with CHHI scores
2. Layer toggles for different metrics
3. Interactive popups with neighborhood details
4. Legend and title
"""

import json
import pandas as pd
import folium
from folium.plugins import Search
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from housing_health.logging_utils import setup_logger, get_timestamped_log_filename

# Paths
DATA_DIR = PROJECT_ROOT / "data"
GEO_DIR = DATA_DIR / "geo"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "interactive"

# Color scales for different risk tiers
RISK_COLORS = {
    'Very High Risk': '#7f0000',  # Dark red
    'High Risk': '#d32f2f',       # Red
    'Elevated Risk': '#ff9800',   # Orange
    'Moderate Risk': '#fdd835',   # Yellow
    'Low Risk': '#4caf50',        # Green
}

# Color scale for continuous values (red = bad, green = good)
def get_color(value, min_val=0, max_val=100):
    """Get color based on value (higher = worse = redder)."""
    if pd.isna(value):
        return '#cccccc'  # Gray for missing
    
    # Normalize to 0-1
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    normalized = max(0, min(1, normalized))
    
    # Color gradient: green (0) -> yellow (0.5) -> red (1)
    if normalized < 0.25:
        # Green to yellow-green
        r = int(76 + (255 - 76) * (normalized / 0.25))
        g = int(175 + (235 - 175) * (normalized / 0.25))
        b = int(80 - 80 * (normalized / 0.25))
    elif normalized < 0.5:
        # Yellow-green to yellow
        r = int(255)
        g = int(235 - (235 - 215) * ((normalized - 0.25) / 0.25))
        b = int(0)
    elif normalized < 0.75:
        # Yellow to orange
        r = int(255)
        g = int(215 - (215 - 152) * ((normalized - 0.5) / 0.25))
        b = int(0)
    else:
        # Orange to red
        r = int(255 - (255 - 211) * ((normalized - 0.75) / 0.25))
        g = int(152 - 152 * ((normalized - 0.75) / 0.25))
        b = int(0)
    
    return f'#{r:02x}{g:02x}{b:02x}'


def load_data(logger):
    """Load neighborhood boundaries and index data."""
    
    # Load GeoJSON boundaries
    geo_files = list(GEO_DIR.glob("nta_boundaries*.geojson"))
    if not geo_files:
        raise FileNotFoundError("No NTA boundaries file found")
    
    geo_file = max(geo_files)  # Most recent
    logger.info(f"Loading boundaries: {geo_file.name}")
    
    with open(geo_file, 'r') as f:
        geojson = json.load(f)
    
    # Load index data
    index_file = PROCESSED_DIR / "housing_health_index.csv"
    if not index_file.exists():
        raise FileNotFoundError("Housing health index file not found")
    
    logger.info(f"Loading index: {index_file.name}")
    df = pd.read_csv(index_file)
    
    # Create lookup by NTA code (drop duplicates, keep first)
    df_unique = df.drop_duplicates(subset=['nta_code'], keep='first')
    index_lookup = df_unique.set_index('nta_code').to_dict('index')
    
    logger.info(f"  Loaded {len(geojson['features'])} boundaries")
    logger.info(f"  Loaded {len(df)} neighborhood indices")
    
    return geojson, df, index_lookup


def create_popup_html(nta_code, nta_name, data):
    """Create HTML content for popup."""
    
    if data is None:
        return f"""
        <div style="font-family: system-ui, -apple-system, sans-serif; min-width: 200px;">
            <h3 style="margin: 0 0 8px 0; color: #333;">{nta_name}</h3>
            <p style="color: #666;">No data available</p>
        </div>
        """
    
    chhi = data.get('housing_health_index', 0)
    risk_tier = data.get('risk_tier', 'Unknown')
    borough = data.get('borough', 'Unknown')
    
    # Get component scores
    violations = data.get('violations_score', 0)
    complaints = data.get('complaints_score', 0)
    asthma = data.get('asthma_score', 0)
    lead = data.get('lead_score', 0)
    
    # Get raw rates
    asthma_rate = data.get('asthma_rate_per_10k', 0)
    lead_rate = data.get('lead_rate_per_1k', 0)
    violation_count = data.get('violation_count', 0)
    complaint_count = data.get('complaint_count', 0)
    
    # Risk tier color
    tier_color = RISK_COLORS.get(risk_tier, '#888')
    
    html = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; min-width: 280px; max-width: 350px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <h3 style="margin: 0; color: #1a1a1a; font-size: 16px;">{nta_name}</h3>
                <span style="color: #666; font-size: 12px;">{borough} · {nta_code}</span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 28px; font-weight: bold; color: {tier_color};">{chhi:.0f}</div>
                <div style="font-size: 10px; color: #666;">CHHI Score</div>
            </div>
        </div>
        
        <div style="background: {tier_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; display: inline-block; margin-bottom: 12px;">
            {risk_tier}
        </div>
        
        <div style="border-top: 1px solid #eee; padding-top: 10px; margin-top: 4px;">
            <div style="font-size: 12px; font-weight: 600; color: #333; margin-bottom: 6px;">Component Scores (percentile)</div>
            <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
                <tr>
                    <td style="padding: 3px 0; color: #666;">Housing Violations</td>
                    <td style="padding: 3px 0; text-align: right; font-weight: 500;">{violations:.0f}</td>
                </tr>
                <tr>
                    <td style="padding: 3px 0; color: #666;">311 Complaints</td>
                    <td style="padding: 3px 0; text-align: right; font-weight: 500;">{complaints:.0f}</td>
                </tr>
                <tr>
                    <td style="padding: 3px 0; color: #666;">Child Asthma Rate</td>
                    <td style="padding: 3px 0; text-align: right; font-weight: 500;">{asthma:.0f}</td>
                </tr>
                <tr>
                    <td style="padding: 3px 0; color: #666;">Lead Poisoning Rate</td>
                    <td style="padding: 3px 0; text-align: right; font-weight: 500;">{lead:.0f}</td>
                </tr>
            </table>
        </div>
        
        <div style="border-top: 1px solid #eee; padding-top: 10px; margin-top: 10px;">
            <div style="font-size: 12px; font-weight: 600; color: #333; margin-bottom: 6px;">Raw Metrics</div>
            <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
                <tr>
                    <td style="padding: 3px 0; color: #666;">Total Violations</td>
                    <td style="padding: 3px 0; text-align: right; font-weight: 500;">{violation_count:,.0f}</td>
                </tr>
                <tr>
                    <td style="padding: 3px 0; color: #666;">Total Complaints</td>
                    <td style="padding: 3px 0; text-align: right; font-weight: 500;">{complaint_count:,.0f}</td>
                </tr>
                <tr>
                    <td style="padding: 3px 0; color: #666;">Asthma ED Visits</td>
                    <td style="padding: 3px 0; text-align: right; font-weight: 500;">{asthma_rate:.0f}/10k</td>
                </tr>
                <tr>
                    <td style="padding: 3px 0; color: #666;">Lead Poisoning</td>
                    <td style="padding: 3px 0; text-align: right; font-weight: 500;">{lead_rate:.1f}/1k</td>
                </tr>
            </table>
        </div>
    </div>
    """
    
    return html


def create_map(geojson, df, index_lookup, logger):
    """Create the Folium choropleth map."""
    
    logger.info("Creating map...")
    
    # NYC center coordinates
    nyc_center = [40.7128, -74.0060]
    
    # Create base map with a clean, light style
    m = folium.Map(
        location=nyc_center,
        zoom_start=10,
        tiles=None,
        prefer_canvas=True
    )
    
    # Add CartoDB Positron as base (light, clean basemap)
    folium.TileLayer(
        tiles='cartodbpositron',
        name='Light Map',
        control=True
    ).add_to(m)
    
    # Add dark option
    folium.TileLayer(
        tiles='cartodbdark_matter',
        name='Dark Map',
        control=True
    ).add_to(m)
    
    # Create feature groups for different layers
    chhi_layer = folium.FeatureGroup(name='Child Health Housing Index', show=True)
    violations_layer = folium.FeatureGroup(name='Violations Score', show=False)
    asthma_layer = folium.FeatureGroup(name='Asthma Score', show=False)
    lead_layer = folium.FeatureGroup(name='Lead Score', show=False)
    
    # Track matched/unmatched
    matched = 0
    unmatched = 0
    
    # Add each neighborhood polygon
    for feature in geojson['features']:
        props = feature['properties']
        nta_code = props.get('nta2020', '')
        nta_name = props.get('ntaname', 'Unknown')
        
        # Skip parks, airports, cemeteries (ntatype != 0)
        if props.get('ntatype', '0') != '0':
            continue
        
        # Get data for this NTA
        data = index_lookup.get(nta_code, None)
        
        if data:
            matched += 1
            chhi = data.get('housing_health_index', 50)
            violations_score = data.get('violations_score', 50)
            asthma_score = data.get('asthma_score', 50)
            lead_score = data.get('lead_score', 50)
        else:
            unmatched += 1
            chhi = None
            violations_score = None
            asthma_score = None
            lead_score = None
        
        # Create popup
        popup_html = create_popup_html(nta_code, nta_name, data)
        popup = folium.Popup(popup_html, max_width=400)
        
        # Add to CHHI layer
        folium.GeoJson(
            feature,
            style_function=lambda x, chhi=chhi: {
                'fillColor': get_color(chhi) if chhi else '#cccccc',
                'color': '#333',
                'weight': 0.5,
                'fillOpacity': 0.7,
            },
            highlight_function=lambda x: {
                'weight': 2,
                'color': '#000',
                'fillOpacity': 0.9,
            },
            tooltip=folium.Tooltip(f"<b>{nta_name}</b><br>CHHI: {chhi:.0f}" if chhi else f"<b>{nta_name}</b>"),
            popup=popup
        ).add_to(chhi_layer)
        
        # Add to violations layer
        folium.GeoJson(
            feature,
            style_function=lambda x, v=violations_score: {
                'fillColor': get_color(v) if v else '#cccccc',
                'color': '#333',
                'weight': 0.5,
                'fillOpacity': 0.7,
            },
            highlight_function=lambda x: {
                'weight': 2,
                'color': '#000',
                'fillOpacity': 0.9,
            },
            tooltip=folium.Tooltip(f"<b>{nta_name}</b><br>Violations: {violations_score:.0f}" if violations_score else f"<b>{nta_name}</b>"),
            popup=popup
        ).add_to(violations_layer)
        
        # Add to asthma layer
        folium.GeoJson(
            feature,
            style_function=lambda x, a=asthma_score: {
                'fillColor': get_color(a) if a else '#cccccc',
                'color': '#333',
                'weight': 0.5,
                'fillOpacity': 0.7,
            },
            highlight_function=lambda x: {
                'weight': 2,
                'color': '#000',
                'fillOpacity': 0.9,
            },
            tooltip=folium.Tooltip(f"<b>{nta_name}</b><br>Asthma: {asthma_score:.0f}" if asthma_score else f"<b>{nta_name}</b>"),
            popup=popup
        ).add_to(asthma_layer)
        
        # Add to lead layer
        folium.GeoJson(
            feature,
            style_function=lambda x, l=lead_score: {
                'fillColor': get_color(l) if l else '#cccccc',
                'color': '#333',
                'weight': 0.5,
                'fillOpacity': 0.7,
            },
            highlight_function=lambda x: {
                'weight': 2,
                'color': '#000',
                'fillOpacity': 0.9,
            },
            tooltip=folium.Tooltip(f"<b>{nta_name}</b><br>Lead: {lead_score:.0f}" if lead_score else f"<b>{nta_name}</b>"),
            popup=popup
        ).add_to(lead_layer)
    
    logger.info(f"  Matched: {matched} neighborhoods")
    logger.info(f"  Unmatched: {unmatched} neighborhoods")
    
    # Add layers to map
    chhi_layer.add_to(m)
    violations_layer.add_to(m)
    asthma_layer.add_to(m)
    lead_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; 
                background: white; 
                padding: 12px 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                z-index: 9999;
                font-family: system-ui, -apple-system, sans-serif;">
        <h3 style="margin: 0 0 4px 0; font-size: 18px; color: #1a1a1a;">NYC Child Health Housing Index</h3>
        <p style="margin: 0; font-size: 12px; color: #666;">Housing conditions & child health outcomes by neighborhood</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 30px; left: 50px; 
                background: white; 
                padding: 12px 16px; 
                border-radius: 8px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                z-index: 9999;
                font-family: system-ui, -apple-system, sans-serif;">
        <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px; color: #333;">Risk Level</div>
        <div style="display: flex; flex-direction: column; gap: 4px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 12px; background: #7f0000; border-radius: 2px;"></div>
                <span style="font-size: 11px; color: #666;">Very High (90+)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 12px; background: #d32f2f; border-radius: 2px;"></div>
                <span style="font-size: 11px; color: #666;">High (75-90)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 12px; background: #ff9800; border-radius: 2px;"></div>
                <span style="font-size: 11px; color: #666;">Elevated (50-75)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 12px; background: #fdd835; border-radius: 2px;"></div>
                <span style="font-size: 11px; color: #666;">Moderate (25-50)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 12px; background: #4caf50; border-radius: 2px;"></div>
                <span style="font-size: 11px; color: #666;">Low (0-25)</span>
            </div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def main():
    """Main function."""
    log_file = get_timestamped_log_filename("create_neighborhood_map")
    logger = setup_logger(__name__, log_file)
    
    logger.info("=" * 60)
    logger.info("STARTING: Create Neighborhood Choropleth Map")
    logger.info("=" * 60)
    
    try:
        # Load data
        geojson, df, index_lookup = load_data(logger)
        
        # Create map
        m = create_map(geojson, df, index_lookup, logger)
        
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save map
        output_file = OUTPUT_DIR / "neighborhood_map.html"
        m.save(str(output_file))
        logger.info(f"Saved map to: {output_file}")
        
        # Also save a simplified version for embedding
        output_file_simple = OUTPUT_DIR / "neighborhood_map_embed.html"
        m.save(str(output_file_simple))
        
        logger.info("=" * 60)
        logger.info("COMPLETED: Neighborhood map created successfully")
        logger.info("=" * 60)
        
        # Print summary
        print(f"\n✅ Map saved to: {output_file}")
        print(f"   Open in browser to view the interactive map")
        
    except Exception as e:
        logger.error(f"Error creating map: {e}")
        raise


if __name__ == "__main__":
    main()

