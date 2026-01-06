"""
NYC Building Violation Lookup
Compare any building to its neighborhood and citywide averages.
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import re
import json
import gzip
import shutil
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from math import radians, cos, sin, asin, sqrt

# Configuration
DB_PATH = Path("data/processed/building_lookup.db")
DB_GZ_PATH = Path("data/processed/building_lookup.db.gz")

# Decompress database on first run if needed
if not DB_PATH.exists() and DB_GZ_PATH.exists():
    with st.spinner("Preparing database (first run only)..."):
        with gzip.open(DB_GZ_PATH, 'rb') as f_in:
            with open(DB_PATH, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

NTA_INDEX_PATH = Path("data/processed/housing_health_index.csv")
NTA_GEO_PATH = Path("data/geo/nta_boundaries_2025-12-26.geojson")
VIOLATIONS_PATH = Path("data/processed/violations_clean.csv")
COMPLAINTS_PATH = Path("data/processed/complaints_clean.csv")

# Color scheme
COLORS = {
    'good': '#16a34a',       # Green - low violations
    'moderate': '#ca8a04',   # Yellow - moderate
    'warning': '#ea580c',    # Orange - elevated
    'danger': '#dc2626',     # Red - high risk
    'neutral': '#6b7280',    # Gray - neutral/unknown
    'muted': '#9ca3af',      # Light gray - secondary text
    'bg_light': '#f9fafb',   # Light background
    'border': '#e5e7eb',     # Border color
}

RISK_TIER_COLORS = {
    'Low Risk': {'bg': '#dcfce7', 'text': '#166534'},
    'Moderate Risk': {'bg': '#fef9c3', 'text': '#854d0e'},
    'Elevated Risk': {'bg': '#ffedd5', 'text': '#c2410c'},
    'High Risk': {'bg': '#fee2e2', 'text': '#991b1b'},
}

# Borough mappings
BORO_MAP = {
    '1': 'Manhattan',
    '2': 'Bronx', 
    '3': 'Brooklyn',
    '4': 'Queens',
    '5': 'Staten Island'
}

BORO_NAME_TO_CODE = {
    'MANHATTAN': '1', 'MN': '1', 'NEW YORK': '1', 'NY': '1',
    'BRONX': '2', 'BX': '2',
    'BROOKLYN': '3', 'BK': '3', 'KINGS': '3',
    'QUEENS': '4', 'QN': '4', 'QNS': '4',
    'STATEN ISLAND': '5', 'SI': '5', 'RICHMOND': '5'
}


from contextlib import contextmanager


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    if not DB_PATH.exists():
        yield None
        return
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def get_db_connection_direct():
    """Direct connection for backward compatibility - prefer context manager."""
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def get_data_update_date():
    """Get the date when the database was last updated."""
    with get_db_connection() as conn:
        if conn is None:
            return None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT created_date FROM summary LIMIT 1")
            result = cursor.fetchone()
            if result and result[0]:
                # Parse and format the date
                date_str = result[0][:10]  # Get YYYY-MM-DD part
                return date_str
        except:
            pass
    return None


def normalize_query(query):
    query = query.upper().strip()
    query = re.sub(r'[,.]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    query = re.sub(r'\b\d{5}(-\d{4})?\b', '', query).strip()
    query = re.sub(r'\bNEW YORK\b', '', query)
    query = re.sub(r'\bNY\b', '', query)
    query = re.sub(r'\s+', ' ', query).strip()

    abbreviations = [
        (r'\bAVE\b', 'AVENUE'), (r'\bST\b', 'STREET'), (r'\bRD\b', 'ROAD'),
        (r'\bBLVD\b', 'BOULEVARD'), (r'\bDR\b', 'DRIVE'), (r'\bPL\b', 'PLACE'),
        (r'\bLN\b', 'LANE'), (r'\bCT\b', 'COURT'), (r'\bPKWY\b', 'PARKWAY'),
        (r'\bPKY\b', 'PARKWAY'), (r'\bTER\b', 'TERRACE'), (r'\bCIR\b', 'CIRCLE'),
    ]
    for pattern, replacement in abbreviations:
        query = re.sub(pattern, replacement, query)
    query = re.sub(r'(\d+)(ST|ND|RD|TH)\b', r'\1', query)
    return query


def get_street_variants(street):
    """Generate common variants of a street name for fuzzy matching."""
    variants = [street]

    # Handle ordinal variations (1 STREET vs FIRST STREET vs 1ST STREET)
    ordinal_words = {
        'FIRST': '1', 'SECOND': '2', 'THIRD': '3', 'FOURTH': '4', 'FIFTH': '5',
        'SIXTH': '6', 'SEVENTH': '7', 'EIGHTH': '8', 'NINTH': '9', 'TENTH': '10',
        'ELEVENTH': '11', 'TWELFTH': '12', 'THIRTEENTH': '13',
    }

    for word, num in ordinal_words.items():
        if word in street:
            variants.append(street.replace(word, num))

    # Try numeric to word
    for word, num in ordinal_words.items():
        if f' {num} ' in f' {street} ':
            variants.append(street.replace(num, word))

    # Handle suffix variations
    suffix_pairs = [
        ('STREET', 'ST'), ('AVENUE', 'AVE'), ('BOULEVARD', 'BLVD'),
        ('ROAD', 'RD'), ('PLACE', 'PL'), ('DRIVE', 'DR'), ('COURT', 'CT'),
        ('LANE', 'LN'), ('PARKWAY', 'PKWY'), ('TERRACE', 'TER'),
    ]
    for full, abbr in suffix_pairs:
        if full in street:
            variants.append(street.replace(full, abbr))
        elif abbr in street:
            variants.append(street.replace(abbr, full))

    # Handle directional variations
    directional_pairs = [
        ('NORTH', 'N'), ('SOUTH', 'S'), ('EAST', 'E'), ('WEST', 'W'),
    ]
    for full, abbr in directional_pairs:
        if f' {full} ' in f' {street} ':
            variants.append(street.replace(full, abbr))
        elif f' {abbr} ' in f' {street} ':
            variants.append(street.replace(abbr, full))

    return list(set(variants))


def parse_address(query):
    query = normalize_query(query)
    boro_code = None
    for boro_name, code in BORO_NAME_TO_CODE.items():
        if query.endswith(' ' + boro_name):
            boro_code = code
            query = query[:-len(boro_name)-1].strip()
            break
        elif ' ' + boro_name + ' ' in query:
            boro_code = code
            query = query.replace(' ' + boro_name + ' ', ' ').strip()
            break
    
    match = re.match(r'^(\d+[-\d]*)\s+(.+)$', query)
    if match:
        return match.group(1), match.group(2), boro_code
    return None, query, boro_code


def search_buildings(query):
    with get_db_connection() as conn:
        if conn is None:
            return [], []

        house_num, street, boro_code = parse_address(query)
        cursor = conn.cursor()
        results = []
        suggestions = []
        col_names = None

        if house_num and street:
            # First try exact match
            sql = "SELECT * FROM buildings WHERE housenumber = ? AND streetname LIKE ?"
            params = [house_num, f'%{street}%']
            if boro_code:
                sql += " AND boro = ?"
                params.append(boro_code)
            sql += " LIMIT 10"
            cursor.execute(sql, params)
            results = cursor.fetchall()
            if cursor.description:
                col_names = [desc[0] for desc in cursor.description]

            # If no results, try street variants (fuzzy matching)
            if not results:
                street_variants = get_street_variants(street)
                for variant in street_variants:
                    if variant == street:
                        continue
                    sql = "SELECT * FROM buildings WHERE housenumber = ? AND streetname LIKE ?"
                    params = [house_num, f'%{variant}%']
                    if boro_code:
                        sql += " AND boro = ?"
                        params.append(boro_code)
                    sql += " LIMIT 10"
                    cursor.execute(sql, params)
                    results = cursor.fetchall()
                    if cursor.description:
                        col_names = [desc[0] for desc in cursor.description]
                    if results:
                        break

            # If still no results, try nearby house numbers
            if not results:
                try:
                    num = int(house_num.split('-')[0])
                    nearby_nums = [str(n) for n in range(num - 10, num + 11)]
                    sql = "SELECT * FROM buildings WHERE housenumber IN ({}) AND streetname LIKE ?".format(
                        ','.join(['?' for _ in nearby_nums]))
                    params = nearby_nums + [f'%{street}%']
                    if boro_code:
                        sql += " AND boro = ?"
                        params.append(boro_code)
                    sql += " LIMIT 20"
                    cursor.execute(sql, params)
                    suggestions = cursor.fetchall()
                    if cursor.description:
                        col_names = [desc[0] for desc in cursor.description]
                except ValueError:
                    pass

        if col_names and (results or suggestions):
            results = [dict(zip(col_names, row)) for row in results]
            suggestions = [dict(zip(col_names, row)) for row in suggestions]

        return results, suggestions


def check_nonresidential_buildings(query):
    """Check if an address exists as a non-residential building."""
    with get_db_connection() as conn:
        if conn is None:
            return None

        house_num, street, boro_code = parse_address(query)
        if not house_num or not street:
            return None

        cursor = conn.cursor()

        # Check if the nonresidential_buildings table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nonresidential_buildings'")
        if not cursor.fetchone():
            return None

        # Try exact match first
        sql = "SELECT * FROM nonresidential_buildings WHERE housenumber = ? AND streetname LIKE ?"
        params = [house_num, f'%{street}%']
        if boro_code:
            sql += " AND boro = ?"
            params.append(boro_code)
        sql += " LIMIT 1"

        cursor.execute(sql, params)
        result = cursor.fetchone()

        if result and cursor.description:
            col_names = [desc[0] for desc in cursor.description]
            return dict(zip(col_names, result))

        # Try street variants
        if not result:
            street_variants = get_street_variants(street)
            for variant in street_variants:
                if variant == street:
                    continue
                sql = "SELECT * FROM nonresidential_buildings WHERE housenumber = ? AND streetname LIKE ?"
                params = [house_num, f'%{variant}%']
                if boro_code:
                    sql += " AND boro = ?"
                    params.append(boro_code)
                sql += " LIMIT 1"
                cursor.execute(sql, params)
                result = cursor.fetchone()
                if result and cursor.description:
                    col_names = [desc[0] for desc in cursor.description]
                    return dict(zip(col_names, result))

        return None


# Building class descriptions for user-friendly messages
BUILDING_CLASS_DESCRIPTIONS = {
    'A': 'One Family Dwelling',
    'B': 'Two Family Dwelling',
    'C': 'Walk-up Apartment',
    'D': 'Elevator Apartment',
    'E': 'Warehouse',
    'F': 'Factory/Industrial',
    'G': 'Garage',
    'H': 'Hotel',
    'I': 'Hospital/Health',
    'J': 'Theater',
    'K': 'Store Building',
    'L': 'Loft Building',
    'M': 'Church/Religious',
    'N': 'Asylum/Home',
    'O': 'Office Building',
    'P': 'Place of Worship',
    'Q': 'Outdoor Recreation',
    'R': 'Condo',
    'S': 'Mixed Residential/Commercial',
    'T': 'Transportation',
    'U': 'Utility',
    'V': 'Vacant Land',
    'W': 'Educational/School',
    'Y': 'Government',
    'Z': 'Miscellaneous',
}


def get_building_class_description(bldgclass):
    """Get a human-readable description for a building class code."""
    if not bldgclass:
        return "Unknown"
    # Take first character of building class
    first_char = str(bldgclass)[0].upper()
    return BUILDING_CLASS_DESCRIPTIONS.get(first_char, f"Class {bldgclass}")


# ============================================================================
# NEW FEATURE: Violation Timeline Data
# ============================================================================

def normalize_bbl(bbl):
    """Normalize BBL to 10-digit string format."""
    if bbl is None:
        return None
    # Convert to string, remove .0 suffix if float
    bbl_str = str(bbl).strip()
    if '.' in bbl_str:
        bbl_str = bbl_str.split('.')[0]
    # Remove scientific notation
    try:
        bbl_int = int(float(bbl))
        return str(bbl_int)
    except (ValueError, TypeError):
        return bbl_str


@st.cache_data(ttl=3600)
def get_violation_timeline(bbl):
    """Get violation history for a building, grouped by month."""
    if not VIOLATIONS_PATH.exists():
        return None

    bbl_normalized = normalize_bbl(bbl)

    # Read only needed columns for this BBL
    try:
        df = pd.read_csv(VIOLATIONS_PATH, usecols=[
            'bbl', 'inspectiondate', 'class', 'currentstatus',
            'is_lead', 'is_mold', 'is_pests', 'is_heat',
            'certifieddate'
        ])
        df['bbl'] = df['bbl'].apply(normalize_bbl)
        df = df[df['bbl'] == bbl_normalized]

        if len(df) == 0:
            return None

        # Parse dates
        df['inspectiondate'] = pd.to_datetime(df['inspectiondate'], errors='coerce')
        df['certifieddate'] = pd.to_datetime(df['certifieddate'], errors='coerce')
        df = df.dropna(subset=['inspectiondate'])

        # Create month column
        df['month'] = df['inspectiondate'].dt.to_period('M').dt.to_timestamp()

        # Calculate resolution time (days to close)
        df['resolution_days'] = (df['certifieddate'] - df['inspectiondate']).dt.days

        return df
    except Exception as e:
        st.error(f"Error loading violation data: {e}")
        return None


def create_violation_timeline_chart(df):
    """Create a plotly timeline chart from violation data."""
    if df is None or len(df) == 0:
        return None

    # Group by month
    monthly = df.groupby('month').agg({
        'class': 'count',
        'is_lead': 'sum',
        'is_mold': 'sum',
        'is_pests': 'sum',
        'is_heat': 'sum'
    }).reset_index()
    monthly.columns = ['Month', 'Total', 'Lead', 'Mold', 'Pests', 'Heat']

    # Create line chart (no fill)
    fig = go.Figure()

    colors = {
        'Total': '#1f2937',
        'Heat': '#2563eb',
        'Pests': '#ea580c',
        'Mold': '#7c3aed',
        'Lead': '#dc2626'
    }

    # Add total line first (thicker)
    fig.add_trace(go.Scatter(
        x=monthly['Month'],
        y=monthly['Total'],
        name='Total',
        mode='lines+markers',
        line=dict(color=colors['Total'], width=2),
        marker=dict(size=4),
    ))

    # Add health category lines
    for col in ['Heat', 'Pests', 'Mold', 'Lead']:
        fig.add_trace(go.Scatter(
            x=monthly['Month'],
            y=monthly[col],
            name=col,
            mode='lines+markers',
            line=dict(color=colors[col], width=1.5),
            marker=dict(size=3),
        ))

    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title='Violations',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=30, b=40),
        height=300,
    )

    return fig


def get_resolution_time_stats(df):
    """Calculate resolution time statistics from violation data."""
    if df is None or len(df) == 0:
        return None

    # Determine open/closed based on currentstatus field
    # "VIOLATION CLOSED" and "VIOLATION DISMISSED" are considered closed
    closed_statuses = ['VIOLATION CLOSED', 'VIOLATION DISMISSED']
    if 'currentstatus' in df.columns:
        df['is_closed'] = df['currentstatus'].str.upper().isin(closed_statuses)
        total_open = (~df['is_closed']).sum()
        total_closed = df['is_closed'].sum()
    else:
        # Fallback: use certifieddate if currentstatus not available
        total_closed = df['certifieddate'].notna().sum()
        total_open = len(df) - total_closed

    # Only look at closed violations with valid resolution days for timing stats
    closed = df[df['resolution_days'].notna() & (df['resolution_days'] >= 0)]

    if len(closed) == 0:
        return {
            'avg_days': None,
            'median_days': None,
            'min_days': None,
            'max_days': None,
            'total_closed': total_closed,
            'total_open': total_open,
        }

    return {
        'avg_days': closed['resolution_days'].mean(),
        'median_days': closed['resolution_days'].median(),
        'min_days': closed['resolution_days'].min(),
        'max_days': closed['resolution_days'].max(),
        'total_closed': total_closed,
        'total_open': total_open,
    }


# ============================================================================
# NEW FEATURE: 311 Complaints Data
# ============================================================================

@st.cache_data(ttl=3600)
def get_complaint_details(bbl):
    """Get detailed complaint history for a building."""
    if not COMPLAINTS_PATH.exists():
        return None

    bbl_normalized = normalize_bbl(bbl)

    try:
        df = pd.read_csv(COMPLAINTS_PATH, usecols=[
            'bbl', 'received_date', 'major_category', 'minor_category',
            'complaint_status', 'is_heat', 'is_mold', 'is_pests', 'is_lead', 'is_water'
        ])
        df['bbl'] = df['bbl'].apply(normalize_bbl)
        df = df[df['bbl'] == bbl_normalized]

        if len(df) == 0:
            return None

        df['received_date'] = pd.to_datetime(df['received_date'], errors='coerce')
        return df
    except Exception as e:
        return None


def get_complaint_summary(df):
    """Summarize complaints by category."""
    if df is None or len(df) == 0:
        return None

    summary = {
        'total': len(df),
        'heat': int(df['is_heat'].sum()),
        'mold': int(df['is_mold'].sum()),
        'pests': int(df['is_pests'].sum()),
        'lead': int(df['is_lead'].sum()),
        'water': int(df['is_water'].sum()),
    }

    # Top categories
    if 'major_category' in df.columns:
        top_cats = df['major_category'].value_counts().head(5).to_dict()
        summary['top_categories'] = top_cats

    return summary


# ============================================================================
# NEW FEATURE: Nearby Buildings
# ============================================================================

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in meters between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r


def get_nearby_buildings(lat, lon, current_bbl, radius_meters=200, limit=10):
    """Find nearby buildings within a radius."""
    with get_db_connection() as conn:
        if conn is None:
            return []

        # Query buildings with coordinates (rough bounding box first for speed)
        lat_delta = radius_meters / 111000  # ~111km per degree latitude
        lon_delta = radius_meters / (111000 * cos(radians(lat)))

        sql = """
        SELECT bbl, housenumber, streetname, boro, latitude, longitude,
               violation_count, violations_open, adjusted_score_pct, risk_tier, unitsres
        FROM buildings
        WHERE latitude BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
          AND bbl != ?
        LIMIT 100
        """

        cursor = conn.cursor()
        cursor.execute(sql, [
            lat - lat_delta, lat + lat_delta,
            lon - lon_delta, lon + lon_delta,
            str(current_bbl)
        ])

        results = cursor.fetchall()
        if not results:
            return []

        col_names = [desc[0] for desc in cursor.description]
        buildings = [dict(zip(col_names, row)) for row in results]

        # Calculate actual distances and filter
        for b in buildings:
            if b['latitude'] and b['longitude']:
                b['distance'] = haversine(lon, lat, b['longitude'], b['latitude'])
            else:
                b['distance'] = float('inf')

        # Filter by actual radius and sort by distance
        nearby = [b for b in buildings if b['distance'] <= radius_meters]
        nearby.sort(key=lambda x: x['distance'])

        return nearby[:limit]


def format_percentile(pct):
    if pct is None or pd.isna(pct):
        return "N/A"
    n = int(round(pct))
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def format_percentile_display(pct, value):
    """Format percentile with special handling for zero values."""
    if pct is None or pd.isna(pct):
        return "N/A"

    n = int(round(pct))
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

    pct_str = f"{n}{suffix}"

    # For zero values, add context
    if value == 0 and n < 30:
        return f"{pct_str} (better than {100-n}%)"

    return pct_str


def get_percentile_color(pct):
    """Get color based on percentile value."""
    if pct is None or pd.isna(pct):
        return COLORS['neutral']
    if pct < 25:
        return COLORS['good']
    elif pct < 50:
        return COLORS['neutral']
    elif pct < 75:
        return COLORS['moderate']
    elif pct < 90:
        return COLORS['warning']
    else:
        return COLORS['danger']


def get_data_update_date():
    """Get the last update date from the database file."""
    try:
        db_path = PROCESSED_DIR / "building_lookup.db"
        if db_path.exists():
            from datetime import datetime
            mtime = db_path.stat().st_mtime
            return datetime.fromtimestamp(mtime).strftime("%b %d, %Y")
    except:
        pass
    return None


def inject_custom_css():
    """Inject custom CSS for improved styling."""
    st.markdown("""
<style>
    /* ========== HERO SECTION WITH NYC SKYLINE ========== */
    .hero-section {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #4a7c9b 100%);
        border-radius: 12px;
        padding: 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 60px;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 120'%3E%3Cpath fill='%23ffffff15' d='M0,120 L0,80 L30,80 L30,60 L50,60 L50,80 L70,80 L70,40 L90,40 L90,80 L120,80 L120,50 L140,50 L140,30 L160,30 L160,50 L180,50 L180,80 L210,80 L210,20 L230,20 L230,80 L260,80 L260,70 L280,70 L280,80 L310,80 L310,45 L330,45 L330,25 L350,25 L350,45 L370,45 L370,80 L400,80 L400,60 L420,60 L420,80 L450,80 L450,35 L470,35 L470,15 L490,15 L490,35 L510,35 L510,80 L540,80 L540,55 L560,55 L560,80 L590,80 L590,40 L610,40 L610,80 L640,80 L640,65 L660,65 L660,80 L690,80 L690,30 L710,30 L710,10 L730,10 L730,30 L750,30 L750,80 L780,80 L780,50 L800,50 L800,80 L830,80 L830,70 L850,70 L850,80 L880,80 L880,45 L900,45 L900,80 L930,80 L930,60 L950,60 L950,80 L980,80 L980,35 L1000,35 L1000,80 L1030,80 L1030,55 L1050,55 L1050,80 L1080,80 L1080,40 L1100,40 L1100,20 L1120,20 L1120,40 L1140,40 L1140,80 L1170,80 L1170,65 L1200,65 L1200,120 Z'/%3E%3C/svg%3E") repeat-x;
        background-size: auto 60px;
        opacity: 0.6;
    }
    .hero-title {
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 16px;
        margin-bottom: 20px;
    }
    .hero-stats {
        display: flex;
        gap: 32px;
        flex-wrap: wrap;
    }
    .hero-stat {
        text-align: center;
    }
    .hero-stat-value {
        color: white;
        font-size: 24px;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .hero-stat-label {
        color: rgba(255,255,255,0.75);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Data source badge */
    .data-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 6px 14px;
        margin-top: 16px;
        color: rgba(255,255,255,0.9);
        font-size: 12px;
    }
    .data-badge img {
        height: 16px;
        opacity: 0.9;
    }

    /* Card styling */
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #374151;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    /* Risk tier badges */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 14px;
        font-weight: 600;
    }

    /* Progress bar container */
    .progress-row {
        display: flex;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #f3f4f6;
    }
    .progress-row:last-child {
        border-bottom: none;
    }
    .progress-label {
        width: 140px;
        font-weight: 500;
        color: #374151;
    }
    .progress-value {
        width: 60px;
        text-align: right;
        font-weight: 600;
        color: #1f2937;
        padding-right: 12px;
    }
    .progress-bar-bg {
        flex: 1;
        background: #e5e7eb;
        border-radius: 4px;
        height: 8px;
        margin: 0 12px;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .progress-pct {
        width: 70px;
        font-size: 13px;
        color: #6b7280;
    }
    .progress-pct-nhood {
        width: 70px;
        font-size: 13px;
        color: #9ca3af;
    }

    /* Zero value styling */
    .zero-value {
        color: #16a34a;
        font-weight: 600;
    }

    /* Severity badges */
    .severity-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin: 4px;
    }
    .severity-a { background: #dcfce7; color: #166534; }
    .severity-b { background: #fef9c3; color: #854d0e; }
    .severity-c { background: #fee2e2; color: #991b1b; }

    /* Trend styling */
    .trend-improving { color: #16a34a; }
    .trend-worsening { color: #dc2626; }
    .trend-stable { color: #6b7280; }

    /* Link button */
    .link-button {
        display: inline-block;
        padding: 8px 16px;
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        color: #374151;
        text-decoration: none;
        font-size: 14px;
    }
    .link-button:hover {
        background: #e5e7eb;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Improve metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }

    /* ========== MOBILE RESPONSIVE ========== */
    @media (max-width: 768px) {
        .hero-section {
            padding: 20px;
        }
        .hero-title {
            font-size: 22px;
        }
        .hero-stats {
            gap: 16px;
        }
        .hero-stat-value {
            font-size: 18px;
        }
        .progress-label {
            width: 100px;
            font-size: 13px;
        }
        .progress-pct, .progress-pct-nhood {
            width: 50px;
            font-size: 11px;
        }
        /* Auto-collapse expanders content on mobile - handled by Streamlit */
        [data-testid="stExpander"] {
            margin-bottom: 8px;
        }
    }
</style>
""", unsafe_allow_html=True)


def display_comparison_bar(label, value, city_pct, nhood_pct):
    """Display a comparison row with visual progress bar."""
    value_int = int(value or 0)
    bar_color = get_percentile_color(city_pct)
    bar_width = min(100, city_pct or 0)

    # Format value display
    if value_int == 0:
        value_display = '<span class="zero-value">0</span>'
    else:
        value_display = f'{value_int}'

    # Format percentile displays
    city_pct_display = format_percentile(city_pct) if city_pct is not None else "N/A"
    nhood_pct_display = format_percentile(nhood_pct) if nhood_pct is not None else "-"

    st.markdown(f"""
    <div class="progress-row">
        <div class="progress-label">{label}</div>
        <div class="progress-value">{value_display}</div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {bar_width}%; background: {bar_color};"></div>
        </div>
        <div class="progress-pct">{city_pct_display}</div>
        <div class="progress-pct-nhood">{nhood_pct_display}</div>
    </div>
    """, unsafe_allow_html=True)


def display_comparison_card(title, subtitle, adjusted_pct, adjusted_label, raw_pct, raw_label, raw_count):
    """Display a comparison card with adjusted and raw percentiles using Streamlit components."""
    adj_color = get_percentile_color(adjusted_pct) if adjusted_pct else COLORS['neutral']
    raw_color = get_percentile_color(raw_pct) if raw_pct else COLORS['neutral']

    adj_pct_int = int(round(adjusted_pct)) if adjusted_pct and not pd.isna(adjusted_pct) else 0
    raw_pct_int = int(round(raw_pct)) if raw_pct and not pd.isna(raw_pct) else 0

    # Use a container with custom styling
    with st.container():
        st.markdown(f"**{title}**")
        st.caption(subtitle)

        # Adjusted Score section
        st.markdown("**Adjusted Score** (per unit)")
        adj_col1, adj_col2 = st.columns([4, 1])
        with adj_col1:
            st.progress(adj_pct_int / 100)
        with adj_col2:
            st.markdown(f"<span style='font-size: 18px; font-weight: 700; color: {adj_color};'>{format_percentile(adjusted_pct)}</span>", unsafe_allow_html=True)
        st.caption(adjusted_label)

        st.markdown("")  # Spacer

        # Raw Violation Count section
        st.markdown("**Raw Violation Count**")
        raw_col1, raw_col2 = st.columns([4, 1])
        with raw_col1:
            st.progress(raw_pct_int / 100)
        with raw_col2:
            st.markdown(f"<span style='font-size: 18px; font-weight: 700; color: {raw_color};'>{format_percentile(raw_pct)}</span>", unsafe_allow_html=True)
        st.caption(f"{raw_label} ({int(raw_count or 0)} violations)")


def display_score_card(title, subtitle, pct, label, count=None):
    """Display a simplified score card with single percentile (used in new UI)."""
    pct_color = get_percentile_color(pct) if pct else COLORS['neutral']
    pct_int = int(round(pct)) if pct and not pd.isna(pct) else 0

    with st.container():
        st.markdown(f"**{title}**")
        st.caption(subtitle)

        col1, col2 = st.columns([4, 1])
        with col1:
            st.progress(pct_int / 100)
        with col2:
            st.markdown(f"<span style='font-size: 24px; font-weight: 700; color: {pct_color};'>{format_percentile(pct)}</span>", unsafe_allow_html=True)

        if count is not None:
            st.caption(f"{label} ({int(count)} total violations)")
        else:
            st.caption(label)


def display_data_table(rows, title=None, caption=None):
    """Display a clean data table with category, count, city %, and neighborhood %."""
    if title:
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)

    # Table header
    st.markdown("""
    <div style="display: flex; padding: 8px 0; border-bottom: 2px solid #e5e7eb; font-weight: 600; color: #6b7280; font-size: 13px;">
        <div style="flex: 2;">Category</div>
        <div style="width: 70px; text-align: right;">Count</div>
        <div style="width: 100px; text-align: right;">vs City</div>
        <div style="width: 100px; text-align: right;">vs Nhood</div>
    </div>
    """, unsafe_allow_html=True)

    for row in rows:
        label = row['label']
        count = int(row.get('count', 0) or 0)
        city_pct = row.get('city_pct')
        nhood_pct = row.get('nhood_pct')
        is_subitem = row.get('subitem', False)

        city_color = get_percentile_color(city_pct) if city_pct else COLORS['neutral']
        nhood_color = get_percentile_color(nhood_pct) if nhood_pct else COLORS['neutral']

        count_display = f'<span style="color: #16a34a; font-weight: 600;">0</span>' if count == 0 else f'<span style="font-weight: 600;">{count}</span>'
        city_display = f'<span style="color: {city_color}; font-weight: 500;">{format_percentile(city_pct)}</span>' if city_pct else '<span style="color: #9ca3af;">-</span>'
        nhood_display = f'<span style="color: {nhood_color}; font-weight: 500;">{format_percentile(nhood_pct)}</span>' if nhood_pct else '<span style="color: #9ca3af;">-</span>'

        indent = "padding-left: 16px; color: #6b7280;" if is_subitem else "font-weight: 500;"
        border = "border-bottom: 1px solid #f3f4f6;" if not is_subitem else ""

        st.markdown(f"""
        <div style="display: flex; padding: 10px 0; {border} align-items: center;">
            <div style="flex: 2; {indent}">{label}</div>
            <div style="width: 70px; text-align: right;">{count_display}</div>
            <div style="width: 100px; text-align: right;">{city_display}</div>
            <div style="width: 100px; text-align: right;">{nhood_display}</div>
        </div>
        """, unsafe_allow_html=True)


def display_building(building, time_period="All time (2019+)"):
    addr = f"{building['housenumber']} {building['streetname']}"
    boro = BORO_MAP.get(str(building['boro']), '')
    year = building.get('yearbuilt')
    year_display = int(year) if year and not pd.isna(year) else 'Unknown'
    units = int(building.get('unitsres', 0) or 0)
    nta = building.get('nta', 'Unknown')
    bbl = building.get('bbl')

    # ========== 1. BUILDING HEADER ==========
    st.markdown(f"## {addr}, {boro}")
    st.caption(f"Built {year_display} | {units} unit{'s' if units != 1 else ''} | {nta}")

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # ========== 2. BUILDING HEALTH SCORE (Hero Section) ==========
    st.markdown('<div class="section-header">Building Health & Safety Score</div>', unsafe_allow_html=True)
    st.caption("Combines HPD violations (severity-weighted), rodent failures, bedbug infestations, and DOB safety violations. Adjusted for building size.")

    adjusted_pct = building.get('adjusted_score_pct')
    violation_count_pct = building.get('violation_count_pct')
    violation_count_nhood_pct = building.get('violation_count_nhood_pct')
    violation_count = building.get('violation_count', 0)

    # For neighborhood comparison, use adjusted score at neighborhood level
    adjusted_nhood_pct = building.get('adjusted_score_nhood_pct')
    if adjusted_nhood_pct is None or pd.isna(adjusted_nhood_pct):
        adjusted_nhood_pct = violation_count_nhood_pct  # Fallback for older data

    # Score type selector - default to adjusted
    score_type = st.selectbox(
        "Score type:",
        ["Comprehensive (adjusted per unit)", "HPD Violations Only (raw count)"],
        index=0,
        help="Comprehensive score includes HPD violations, rodent failures, bedbug infestations, and DOB violations - adjusted for building size. HPD Only shows raw violation count."
    )

    use_adjusted = score_type == "Comprehensive (adjusted per unit)"

    col1, col2 = st.columns(2)

    with col1:
        if use_adjusted:
            pct_display = adjusted_pct
            if units > 1:
                label = f"Worse than {int(adjusted_pct or 0)}% of NYC buildings (adjusted for {units} units)"
            else:
                label = f"Worse than {int(adjusted_pct or 0)}% of NYC buildings"
        else:
            pct_display = violation_count_pct
            label = f"More HPD violations than {int(violation_count_pct or 0)}% of NYC buildings"

        display_score_card(
            title="vs All NYC Buildings",
            subtitle="767,000+ residential buildings",
            pct=pct_display,
            label=label,
            count=violation_count if not use_adjusted else None
        )

    with col2:
        if use_adjusted:
            pct_display_nhood = adjusted_nhood_pct
            label_nhood = f"Worse than {int(adjusted_nhood_pct or 0)}% of buildings in {nta}"
        else:
            pct_display_nhood = violation_count_nhood_pct
            label_nhood = f"More HPD violations than {int(violation_count_nhood_pct or 0)}% locally"

        display_score_card(
            title=f"vs {nta}",
            subtitle="Buildings in this neighborhood",
            pct=pct_display_nhood,
            label=label_nhood,
            count=violation_count if not use_adjusted else None
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # ========== 3. VIOLATIONS CONTRIBUTING TO SCORE ==========
    st.markdown('<div class="section-header">HPD Housing Violations</div>', unsafe_allow_html=True)
    st.caption('Contributes to score. [Source: NYC HPD](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5)')

    # Main violation metrics
    total_violations = int(building.get('violation_count', 0) or 0)
    open_violations = int(building.get('violations_open', 0) or 0)
    class_c = int(building.get('class_c_count', 0) or 0)
    class_b = int(building.get('class_b_count', 0) or 0)
    class_a = int(building.get('class_a_count', 0) or 0)
    class_i = int(building.get('class_i_count', 0) or 0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total HPD Violations", total_violations)
        st.caption(f"{format_percentile(building.get('violation_count_pct'))} percentile citywide")
    with col2:
        st.metric("Open Violations", open_violations)
        st.caption(f"{format_percentile(building.get('violations_open_pct'))} percentile citywide")

    # Class breakdown in expander - these sum to total
    with st.expander("By Severity Class"):
        st.caption("**These sum to Total.** Class C (Immediately Hazardous) = 3x weight, Class B (Hazardous) = 2x, Class A (Non-Hazardous) = 1x, Class I (Orders) = 2.5x")
        class_rows = [
            {'label': 'Class C (Immediately Hazardous)', 'count': class_c,
             'city_pct': building.get('class_c_count_pct'), 'nhood_pct': None},
            {'label': 'Class B (Hazardous)', 'count': class_b,
             'city_pct': building.get('class_b_count_pct'), 'nhood_pct': None},
            {'label': 'Class A (Non-Hazardous)', 'count': class_a,
             'city_pct': building.get('class_a_count_pct'), 'nhood_pct': None},
            {'label': 'Class I (Orders/Vacate)', 'count': class_i,
             'city_pct': building.get('class_i_count_pct'), 'nhood_pct': None},
        ]
        display_data_table(class_rows)

    # Health categories in expander - these are a different slicing
    lead = int(building.get('lead_violations', 0) or 0)
    mold = int(building.get('mold_violations', 0) or 0)
    pest = int(building.get('pest_violations', 0) or 0)
    heat = int(building.get('heat_violations', 0) or 0)

    with st.expander("By Health Category"):
        st.caption("**Different slicing by keyword.** May overlap - a single violation can be flagged for multiple categories.")
        health_rows = [
            {'label': 'Lead Violations', 'count': lead,
             'city_pct': building.get('lead_violations_pct'), 'nhood_pct': building.get('lead_violations_nhood_pct')},
            {'label': 'Mold Violations', 'count': mold,
             'city_pct': building.get('mold_violations_pct'), 'nhood_pct': building.get('mold_violations_nhood_pct')},
            {'label': 'Pest Violations', 'count': pest,
             'city_pct': building.get('pest_violations_pct'), 'nhood_pct': building.get('pest_violations_nhood_pct')},
            {'label': 'Heat/Hot Water', 'count': heat,
             'city_pct': building.get('heat_violations_pct'), 'nhood_pct': building.get('heat_violations_nhood_pct')},
        ]
        display_data_table(health_rows)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # ========== 3.5 ADDITIONAL HEALTH & SAFETY DATA ==========
    rodent_failures = int(building.get('rodent_failures', 0) or 0)
    rodent_inspections = int(building.get('rodent_inspections', 0) or 0)
    bedbug_reports = int(building.get('bedbug_reports', 0) or 0)
    bedbug_units = int(building.get('bedbug_infested_units', 0) or 0)
    dob_violations = int(building.get('dob_health_violations', 0) or 0)
    dob_fines = float(building.get('dob_total_fines', 0) or 0)

    # Show section if any data exists
    has_additional_data = (rodent_inspections > 0 or bedbug_reports > 0 or dob_violations > 0)

    if has_additional_data:
        st.markdown('<div class="section-header">Additional Health & Safety Data</div>', unsafe_allow_html=True)
        st.caption("These also contribute to the Building Health & Safety Score above.")

        # Rodent Inspections
        if rodent_inspections > 0 or rodent_failures > 0:
            st.markdown("**Rodent Inspections**")
            st.caption('[Source: DOHMH](https://data.cityofnewyork.us/Health/Rodent-Inspection/p937-wjvj)')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Failed Inspections", rodent_failures)
            with col2:
                st.metric("Total Inspections", rodent_inspections)
            with col3:
                if rodent_inspections > 0:
                    fail_rate = (rodent_failures / rodent_inspections) * 100
                    st.metric("Failure Rate", f"{fail_rate:.0f}%")
            st.caption(f"{format_percentile(building.get('rodent_failures_pct'))} percentile citywide")
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # Bedbug Reports
        if bedbug_reports > 0 or bedbug_units > 0:
            st.markdown("**Bedbug Infestations**")
            st.caption('[Source: HPD Bedbug Reporting](https://data.cityofnewyork.us/Housing-Development/Bedbug-Reporting/wz6d-d3jb)')
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Filing Reports", bedbug_reports)
            with col2:
                st.metric("Units Reported Infested", bedbug_units)
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # DOB Safety Violations
        if dob_violations > 0 or dob_fines > 0:
            st.markdown("**DOB Safety Violations**")
            st.caption("Structural, fire safety, asbestos. [Source: DOB ECB](https://data.cityofnewyork.us/Housing-Development/DOB-ECB-Violations/6bgk-3dad)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Health/Safety Violations", dob_violations)
            with col2:
                hazardous = int(building.get('dob_hazardous_count', 0) or 0)
                st.metric("Hazardous", hazardous)
            with col3:
                if dob_fines > 0:
                    st.metric("Total Fines", f"${dob_fines:,.0f}")
                else:
                    st.metric("Total Fines", "$0")

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # ========== 4. COMPLAINTS SUMMARY ==========
    complaint_rows = [
        {'label': 'Total Complaints', 'count': building.get('complaint_count', 0),
         'city_pct': building.get('complaints_pct'), 'nhood_pct': building.get('complaints_nhood_pct')},
    ]

    # Get complaint details for category breakdown
    complaint_data = get_complaint_details(bbl)
    complaint_summary = get_complaint_summary(complaint_data)

    if complaint_summary:
        complaint_rows.extend([
            {'label': 'Heat/Hot Water', 'count': complaint_summary.get('heat', 0),
             'city_pct': None, 'nhood_pct': None, 'subitem': True},
            {'label': 'Water/Leaks', 'count': complaint_summary.get('water', 0),
             'city_pct': None, 'nhood_pct': None, 'subitem': True},
            {'label': 'Pests', 'count': complaint_summary.get('pests', 0),
             'city_pct': None, 'nhood_pct': None, 'subitem': True},
            {'label': 'Mold', 'count': complaint_summary.get('mold', 0),
             'city_pct': None, 'nhood_pct': None, 'subitem': True},
            {'label': 'Lead', 'count': complaint_summary.get('lead', 0),
             'city_pct': None, 'nhood_pct': None, 'subitem': True},
        ])

    display_data_table(complaint_rows,
                       title="311 Complaints (For Reference)",
                       caption="Tenant-reported issues via 311. NOT included in score (subjective, hard to normalize). [Source: HPD Complaints](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Complaints/uwyv-629c)")

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # ========== 5. TREND ANALYSIS ==========
    violations_1yr = int(building.get('violations_1yr', 0) or 0)
    violations_2yr = int(building.get('violations_2yr', 0) or 0)
    prior_year = violations_2yr - violations_1yr

    if violations_1yr > 0 or prior_year > 0:
        st.markdown('<div class="section-header">Violation Trend</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Last 12 Months", violations_1yr)
        with col2:
            st.metric("Prior 12 Months", prior_year)
        with col3:
            if prior_year > 0:
                change_pct = ((violations_1yr - prior_year) / prior_year) * 100
                if change_pct < -20:
                    trend_color = "#16a34a"
                    trend_icon = "↓"
                    trend_text = f"{abs(change_pct):.0f}% fewer"
                    trend_label = "Improving"
                elif change_pct > 20:
                    trend_color = "#dc2626"
                    trend_icon = "↑"
                    trend_text = f"{change_pct:.0f}% more"
                    trend_label = "Worsening"
                else:
                    trend_color = "#6b7280"
                    trend_icon = "→"
                    trend_text = "Similar"
                    trend_label = "Stable"

                st.markdown(f"""
                <div style="background: {trend_color}15; border: 1px solid {trend_color};
                            border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 24px; color: {trend_color}; font-weight: 700;">
                        {trend_icon} {trend_label}
                    </div>
                    <div style="font-size: 12px; color: #6b7280;">{trend_text} vs prior year</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("No prior year data for comparison")

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # ========== 6. VIOLATION TIMELINE (Expandable) ==========
    violation_data = get_violation_timeline(bbl)

    if violation_data is not None and len(violation_data) > 0:
        with st.expander("Violation Timeline & Resolution"):
            st.caption("Monthly violation trends and resolution statistics")

            fig = create_violation_timeline_chart(violation_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Resolution time stats
            resolution_stats = get_resolution_time_stats(violation_data)
            if resolution_stats:
                st.markdown("**Resolution Time** (how quickly violations are addressed)")
                col1, col2, col3, col4 = st.columns(4)
                avg_days = resolution_stats.get('avg_days')
                median_days = resolution_stats.get('median_days')
                with col1:
                    if avg_days is not None:
                        st.metric("Avg Days to Close", f"{avg_days:.0f}")
                    else:
                        st.metric("Avg Days to Close", "N/A")
                with col2:
                    if median_days is not None:
                        st.metric("Median Days", f"{median_days:.0f}")
                    else:
                        st.metric("Median Days", "N/A")
                with col3:
                    st.metric("Closed", resolution_stats['total_closed'])
                with col4:
                    st.metric("Still Open", resolution_stats['total_open'])

                if avg_days is not None:
                    if avg_days > 90:
                        st.caption("This building takes longer than average to resolve violations")
                    elif avg_days < 30:
                        st.caption("This building resolves violations relatively quickly")

    # ========== 7. NEARBY BUILDINGS (Expandable) ==========
    lat = building.get('latitude')
    lon = building.get('longitude')

    if lat and lon and not pd.isna(lat) and not pd.isna(lon):
        with st.expander("Nearby Buildings Comparison"):
            st.caption("Other residential buildings within 200 meters")

            nearby = get_nearby_buildings(lat, lon, bbl, radius_meters=200, limit=8)

            if nearby:
                # Calculate comparison
                avg_violations = sum(b.get('violation_count', 0) or 0 for b in nearby) / len(nearby)
                this_violations = int(building.get('violation_count', 0) or 0)
                # Count buildings with FEWER violations than this one
                has_fewer = sum(1 for b in nearby if (b.get('violation_count', 0) or 0) < this_violations)
                # Count buildings with MORE violations than this one
                has_more = sum(1 for b in nearby if (b.get('violation_count', 0) or 0) > this_violations)

                # Show summary
                if this_violations > avg_violations * 1.5:
                    st.warning(f"This building has **more violations** ({this_violations}) than {has_fewer} of {len(nearby)} nearby buildings (avg: {avg_violations:.0f})")
                elif this_violations < avg_violations * 0.5:
                    st.success(f"This building has **fewer violations** ({this_violations}) than {has_more} of {len(nearby)} nearby buildings (avg: {avg_violations:.0f})")
                else:
                    st.info(f"This building has **similar violations** ({this_violations}) to nearby buildings (avg: {avg_violations:.0f})")

                # Create a comparison table
                nearby_data = []
                for b in nearby:
                    nearby_data.append({
                        'Address': f"{b['housenumber']} {b['streetname']}",
                        'Units': int(b.get('unitsres', 0) or 0),
                        'Violations': int(b.get('violation_count', 0) or 0),
                        'Open': int(b.get('violations_open', 0) or 0),
                        'Score': f"{int(b.get('adjusted_score_pct', 0) or 0)}th",
                        'Distance': f"{int(b['distance'])}m",
                    })

                nearby_df = pd.DataFrame(nearby_data)
                st.dataframe(nearby_df, hide_index=True, use_container_width=True)
            else:
                st.caption("No nearby buildings found in database")

    # ========== 8. ACTION BUTTONS ==========
    st.markdown('<div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #e5e7eb;"></div>', unsafe_allow_html=True)

    compare_count = len(st.session_state.get('compare_buildings', []))
    already_added = bbl in [b.get('bbl') for b in st.session_state.get('compare_buildings', [])]

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if already_added:
            st.markdown('<span style="color: #16a34a; font-weight: 500;">Added to comparison</span>', unsafe_allow_html=True)
        elif compare_count >= 3:
            st.caption("Max 3 buildings in comparison")
        else:
            if st.button("Add to Compare", key=f"compare_{bbl}"):
                add_to_compare(building)
                st.rerun()

    with col2:
        building_id = building.get('buildingid')
        if building_id and not pd.isna(building_id):
            hpd_url = f"https://hpdonline.nyc.gov/hpdonline/building/{int(building_id)}"
            st.markdown(f'<a href="{hpd_url}" target="_blank" class="link-button">View on HPD</a>', unsafe_allow_html=True)


@st.cache_data
def load_neighborhood_data():
    if not NTA_INDEX_PATH.exists():
        return None
    return pd.read_csv(NTA_INDEX_PATH)


@st.cache_data
def load_geojson():
    if not NTA_GEO_PATH.exists():
        return None
    with open(NTA_GEO_PATH, 'r') as f:
        return json.load(f)


def display_neighborhood_map():
    try:
        import folium
        from streamlit_folium import st_folium
        import branca.colormap as cm
    except ImportError:
        st.error("Map requires folium, streamlit-folium, and branca. Install with: pip install folium streamlit-folium branca")
        return

    try:
        nta_data = load_neighborhood_data()
        geojson = load_geojson()
    except Exception as e:
        st.error(f"Error loading neighborhood data: {str(e)}")
        return

    if nta_data is None:
        st.warning("Neighborhood data CSV not found at expected path.")
        st.caption(f"Expected: {NTA_INDEX_PATH}")
        return

    if geojson is None:
        st.warning("GeoJSON boundary file not found.")
        st.caption(f"Expected: {NTA_GEO_PATH}")
        return

    # Validate GeoJSON structure
    if 'features' not in geojson or not geojson['features']:
        st.error("GeoJSON file is empty or malformed.")
        return

    metric_options = {
        'Violations per Building': 'violation_per_bldg',
        'Complaints per Building': 'complaint_per_bldg',
        'Health Violations per Building': 'violation_health_per_bldg',
        'Housing Health Index': 'housing_health_index',
        'Lead Violations per 1000 Residents': 'lead_rate_per_1k',
    }

    selected_metric = st.selectbox("Select metric:", list(metric_options.keys()))
    metric_col = metric_options[selected_metric]

    # Create a lookup dict for the metric values
    nta_to_data = {}
    for _, row in nta_data.iterrows():
        nta_code = row.get('nta_code', '')
        nta_to_data[nta_code] = {
            'name': row.get('nta', 'Unknown'),
            'borough': row.get('borough', ''),
            'value': row.get(metric_col, 0),
            'buildings': int(row.get('building_count', 0)),
            'risk_tier': row.get('risk_tier', 'Unknown'),
        }

    # Get min/max for color scale
    valid_values = nta_data[metric_col].dropna()
    if len(valid_values) == 0:
        st.warning("No data available for this metric.")
        return

    vmin, vmax = valid_values.min(), valid_values.max()

    # Create colormap - using colorblind-friendly palette
    colormap = cm.LinearColormap(
        colors=['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'],
        vmin=vmin, vmax=vmax,
        caption=selected_metric
    )

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles='cartodbpositron')

    # Style function for GeoJson
    def style_function(feature):
        nta_code = feature['properties'].get('nta2020', '')
        data = nta_to_data.get(nta_code, {})
        value = data.get('value', 0)
        if pd.isna(value) or value == 0:
            return {'fillColor': '#cccccc', 'color': '#666666', 'weight': 1, 'fillOpacity': 0.4}
        return {
            'fillColor': colormap(value),
            'color': '#333333',
            'weight': 1,
            'fillOpacity': 0.7
        }

    # Highlight function for hover
    def highlight_function(feature):
        return {'fillOpacity': 0.9, 'weight': 3, 'color': '#333333'}

    # Add GeoJson with tooltips
    geojson_layer = folium.GeoJson(
        geojson,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['ntaname'],
            aliases=['Neighborhood:'],
            style='font-size: 12px; padding: 5px;',
            sticky=True
        )
    )

    # Add custom tooltips with data
    for feature in geojson['features']:
        nta_code = feature['properties'].get('nta2020', '')
        data = nta_to_data.get(nta_code, {})
        if data:
            feature['properties']['metric_value'] = f"{data.get('value', 0):.1f}"
            feature['properties']['buildings'] = str(data.get('buildings', 0))
            feature['properties']['risk_tier'] = data.get('risk_tier', 'Unknown')

    # Re-create with updated properties
    geojson_layer = folium.GeoJson(
        geojson,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['ntaname', 'metric_value', 'buildings', 'risk_tier'],
            aliases=['Neighborhood:', f'{selected_metric}:', 'Buildings:', 'Risk Tier:'],
            style='font-size: 12px; padding: 8px; background-color: white;',
            sticky=True
        )
    )

    geojson_layer.add_to(m)
    colormap.add_to(m)

    st_folium(m, width=800, height=500, returned_objects=[])

    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Highest (Most Issues)**")
        top = nta_data.nlargest(10, metric_col)[['nta', 'borough', metric_col, 'risk_tier']]
        top.columns = ['Neighborhood', 'Borough', selected_metric, 'Risk Tier']
        st.dataframe(top, hide_index=True)

    with col2:
        st.write("**Lowest (Fewest Issues)**")
        bottom = nta_data.nsmallest(10, metric_col)[['nta', 'borough', metric_col, 'risk_tier']]
        bottom.columns = ['Neighborhood', 'Borough', selected_metric, 'Risk Tier']
        st.dataframe(bottom, hide_index=True)


def set_search(addr):
    st.session_state['search_query_key'] = addr


def add_to_compare(building):
    """Add a building to the comparison list."""
    if 'compare_buildings' not in st.session_state:
        st.session_state['compare_buildings'] = []

    bbl = building.get('bbl')
    if bbl and bbl not in [b.get('bbl') for b in st.session_state['compare_buildings']]:
        if len(st.session_state['compare_buildings']) < 3:
            st.session_state['compare_buildings'].append(building)
            return True
    return False


def remove_from_compare(bbl):
    """Remove a building from the comparison list."""
    if 'compare_buildings' in st.session_state:
        st.session_state['compare_buildings'] = [
            b for b in st.session_state['compare_buildings'] if b.get('bbl') != bbl
        ]


def display_comparison_view():
    """Display side-by-side comparison of buildings."""
    buildings = st.session_state.get('compare_buildings', [])
    if len(buildings) < 2:
        st.info("Add 2-3 buildings to compare them side-by-side. Search for buildings in the Building Lookup tab.")
        return

    st.markdown('<div class="section-header">Building Comparison</div>', unsafe_allow_html=True)

    # Create columns for each building header
    cols = st.columns(len(buildings))

    for i, (col, b) in enumerate(zip(cols, buildings)):
        with col:
            addr = f"{b.get('housenumber', '')} {b.get('streetname', '')}"
            boro = BORO_MAP.get(str(b.get('boro', '')), '')
            risk_tier = b.get('risk_tier', 'Unknown')
            tier_style = RISK_TIER_COLORS.get(risk_tier, {'bg': '#f3f4f6', 'text': '#6b7280'})

            st.markdown(f"""
            <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                <div style="font-weight: 600; font-size: 15px;">{addr}</div>
                <div style="font-size: 13px; color: #6b7280;">{boro} | {b.get('nta', 'Unknown')}</div>
                <span class="risk-badge" style="background: {tier_style['bg']}; color: {tier_style['text']}; font-size: 12px; margin-top: 8px;">
                    {risk_tier}
                </span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Remove", key=f"remove_{b.get('bbl')}"):
                remove_from_compare(b.get('bbl'))
                st.rerun()

    # Comparison metrics table
    metrics = [
        ('Adjusted Score', 'weighted_score', 'adjusted_score_pct'),  # Primary metric
        ('Total Violations', 'violation_count', 'violation_count_pct'),
        ('Open Violations', 'violations_open', 'violations_open_pct'),
        ('Class C (Hazardous)', 'class_c_count', 'class_c_count_pct'),
        ('Lead Violations', 'lead_violations', 'lead_violations_pct'),
        ('Complaints', 'complaint_count', 'complaints_pct'),
        ('Year Built', 'yearbuilt', None),
        ('Units', 'unitsres', None),
        ('Score/Unit', 'weighted_score_per_unit', 'weighted_score_per_unit_pct'),
    ]

    st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)

    for label, val_col, pct_col in metrics:
        cols = st.columns([1.2] + [1] * len(buildings))
        with cols[0]:
            st.markdown(f'<span style="font-weight: 500; color: #374151;">{label}</span>', unsafe_allow_html=True)

        for i, b in enumerate(buildings):
            with cols[i + 1]:
                val = b.get(val_col, 0) or 0
                pct = b.get(pct_col) if pct_col else None

                if isinstance(val, float) and val == int(val):
                    val = int(val)
                elif isinstance(val, float):
                    val = f"{val:.1f}"

                if pct is not None:
                    color = get_percentile_color(pct)
                    st.markdown(f'<span style="font-weight: 600;">{val}</span> <span style="color: {color}; font-size: 13px;">({format_percentile(pct)})</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span style="font-weight: 600;">{val}</span>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
    if st.button("Clear Comparison"):
        st.session_state['compare_buildings'] = []
        st.rerun()


def display_about_page():
    """Display the About & Methodology page."""

    st.markdown("## About This Tool")
    st.markdown("""
    This tool helps NYC residents, tenant advocates, and researchers understand housing conditions
    at the building level. By combining multiple city data sources, it provides a comprehensive
    view of building health that goes beyond any single dataset.
    """)

    st.markdown("---")

    # Data Sources
    st.markdown("## Data Sources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **HPD Violations** (Primary)
        - Source: [NYC Open Data](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5)
        - ~5.5 million violation records
        - Includes: inspection date, violation class, description, status
        - Updated: Weekly

        **HPD Complaints**
        - Source: [NYC Open Data](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Complaints/uwyv-629c)
        - ~3.7 million complaint records
        - Tenant-reported issues before inspection
        - Categories: heat, water, pests, mold, lead
        """)

    with col2:
        st.markdown("""
        **PLUTO (Building Data)**
        - Source: [NYC Open Data](https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks)
        - Building characteristics for all NYC
        - Includes: year built, units, building class
        - Used for per-unit normalization

        **DOHMH Rodent Inspections**
        - Source: [NYC Open Data](https://data.cityofnewyork.us/Health/Rodent-Inspection/p937-wjvj)
        - Health Department rat inspections
        - Pass/fail results by building
        """)

    st.markdown("---")

    # Methodology
    st.markdown("## Methodology")

    st.markdown("### Comparison Metrics")
    st.markdown("""
    We show two key metrics to help you understand how a building compares:

    **1. Adjusted Score (Primary Metric)**

    A fair comparison metric that accounts for building size and violation severity:
    - **Severity Weighting**: Class C (Immediately Hazardous) = 3x, Class B (Hazardous) = 2x,
      Class A (Non-Hazardous) = 1x, Class I (Orders) = 2.5x, Rodent Failures = 2x
    - **Per-Unit Normalization**: For buildings with 2+ units, the weighted score is divided
      by the number of units. A 100-unit building with 100 violations is treated the same as
      a 10-unit building with 10 violations.
    - **Percentile**: Ranked against all ~767,000 NYC residential buildings

    **2. Raw Violation Count**

    The simple total number of violations, without any adjustments:
    - Useful for understanding the absolute scale of issues
    - Larger buildings naturally have more violations
    - Compare with Adjusted Score to see if a high count is due to building size

    **Why Both?** A large apartment building might rank high on Raw Count (94th percentile)
    but moderate on Adjusted Score (60th percentile) — meaning it has many violations but
    that's typical for its size. Conversely, a small building with high Adjusted Score but
    low Raw Count has serious issues relative to its size.
    """)

    st.markdown("### Health-Related Violation Detection")
    st.markdown("""
    Violations are categorized as health-related using keyword matching on violation descriptions:

    | Category | Keywords |
    |----------|----------|
    | **Lead** | lead, lead-based, lead paint, abatement, xrf, chipping paint, peeling paint |
    | **Mold** | mold, mildew, fungus, moisture, water damage, dampness, leak |
    | **Pests** | roach, cockroach, bedbug, rodent, mice, rat, pest, infestation |
    | **Heat** | heat, no heat, heating, hot water, boiler, radiator, temperature |
    """)

    st.markdown("### Resolution Time")
    st.markdown("""
    Resolution time measures how quickly a building's violations are addressed:
    - Calculated as: `Certified Date - Inspection Date`
    - Only includes closed violations with valid dates
    - **Average < 30 days**: Building resolves issues quickly
    - **Average > 90 days**: Building is slow to resolve issues
    """)

    st.markdown("### Nearby Buildings Comparison")
    st.markdown("""
    The nearby buildings feature uses the Haversine formula to find residential buildings
    within 200 meters. This provides local context - is this building typical for its block,
    or an outlier?
    """)

    st.markdown("---")

    # Limitations
    st.markdown("## Limitations & Caveats")

    st.warning("""
    **Important:** This tool is for informational purposes only. Please consider these limitations:
    """)

    st.markdown("""
    1. **Data Currency**: Violation data reflects historical records. Our database is rebuilt
       periodically but may not reflect violations added or closed in the last few days.

    2. **Open vs Closed Discrepancies**: Our "open violations" count may differ from HPD Online
       because we capture a snapshot, while HPD shows real-time status.

    3. **Non-Residential Buildings**: This tool only includes buildings with 1+ residential
       units. Commercial, industrial, and vacant buildings are excluded.

    4. **Keyword Matching**: Health-related violation categorization uses keyword matching,
       which may miss some relevant violations or incorrectly categorize others.

    5. **311 Complaints vs HPD Violations**: Complaints are tenant-reported issues; violations
       are inspector-confirmed. A building with many complaints but few violations may indicate
       responsive management (issues fixed before inspection).

    6. **Correlation ≠ Causation**: Housing conditions correlate with but don't necessarily
       cause health outcomes. Many factors affect child health.
    """)

    st.markdown("---")

    # How to Use
    st.markdown("## How to Use This Tool")

    st.markdown("""
    **For Tenants/Prospective Renters:**
    - Search for a building before signing a lease
    - Compare to nearby buildings to see if issues are building-specific or area-wide
    - Check resolution time to gauge landlord responsiveness
    - Look at the timeline to see if conditions are improving or worsening

    **For Tenant Advocates:**
    - Identify buildings with persistent issues (high violation counts, slow resolution)
    - Use the neighborhood map to identify problem areas
    - Compare multiple buildings side-by-side

    **For Researchers:**
    - Explore correlations between housing conditions and health outcomes
    - Analyze neighborhood-level patterns
    - Study building-level risk factors
    """)

    st.markdown("---")

    # Technical Details
    with st.expander("Technical Details"):
        st.markdown("""
        **Database Statistics:**
        - Total residential buildings: 767,210
        - Buildings with violations: ~180,000 (23%)
        - Buildings with complaints: ~113,000 (15%)
        - Non-residential buildings (for lookup feedback): 91,074

        **Processing Pipeline:**
        1. Data collection from NYC Open Data APIs
        2. Cleaning and standardization (dates, BBLs, addresses)
        3. Health violation keyword flagging
        4. Spatial NTA assignment using geopandas
        5. Aggregation to building level
        6. Percentile calculation
        7. SQLite database creation for fast lookups

        **Technology Stack:**
        - Python 3.11+
        - Streamlit (web framework)
        - Pandas (data processing)
        - Geopandas (spatial joins)
        - SQLite (database)
        - Plotly (charts)
        - Folium (maps)

        **Source Code:**
        [GitHub Repository](https://github.com/yoelplutchok/nyc-housing-health)

        **Data Freshness:**
        Our database is rebuilt periodically from NYC Open Data snapshots. Violation totals may differ
        slightly from HPD Online due to timing differences. Open violation counts are generally accurate
        within a few days of HPD's real-time data.
        """)

    # Data freshness
    update_date = get_data_update_date()
    if update_date:
        st.caption(f"Database last updated: {update_date}")


def main():
    st.set_page_config(
        page_title="NYC Building Comparison",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Inject custom CSS
    inject_custom_css()

    # Sidebar with data context
    with st.sidebar:
        st.markdown("### Data Information")
        update_date = get_data_update_date()
        st.caption(f"Last updated: {update_date if update_date else 'Unknown'}")
        st.caption("Source: NYC HPD Open Data")

        st.markdown("---")
        st.markdown("**Understanding the Data**")
        st.markdown("""
- **This app:** All violations filed 2019-present (historical record)
- **HPD Online:** Only currently open violations (real-time status)
- A building with many past violations (now resolved) will show high counts here but may show few on HPD Online
        """)

        st.markdown("**Adjusted Health Score:**")
        st.markdown("""
The primary "Adjusted Health Score" factors in:
- **Severity:** Class C (3x), Class B (2x), Class A (1x)
- **Status:** Open violations (2x), Closed (1x)
- **Recency:** Last year (100%), 1-2yr ago (70%), Older (40%)
- **Building size:** Multi-unit buildings use per-unit scores
        """)

        st.markdown("**What percentiles mean:**")
        st.markdown("""
- Higher percentile = more issues than most buildings
- 90th percentile = more issues than 90% of NYC buildings
        """)

        st.markdown("---")
        st.markdown("**Data Sources**")
        st.caption("HPD Housing Maintenance Code Violations")
        st.caption("HPD Complaints")
        st.caption("DOHMH Rodent Inspections")
        st.caption("Bedbug Reporting")
        st.caption("PLUTO (building characteristics)")

        st.markdown("---")
        st.markdown("**Limitations**")
        st.caption("Data may lag HPD Online by days/weeks")
        st.caption("Violation categorization uses keyword matching")
        st.caption("Pre-2019 violations are not included")

    # ========== HERO SECTION ==========
    update_date = get_data_update_date()
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">NYC Building Health & Safety Index</div>
        <div class="hero-subtitle">Compare any building to 767,000+ residential properties across New York City</div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-value">767,210</div>
                <div class="hero-stat-label">Buildings Tracked</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">5.5M</div>
                <div class="hero-stat-label">Violations Analyzed</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">2.2M</div>
                <div class="hero-stat-label">311 Complaints</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">262</div>
                <div class="hero-stat-label">Neighborhoods</div>
            </div>
        </div>
        <div class="data-badge">
            <span>📊</span>
            <span>Powered by NYC Open Data • Updated {update_date if update_date else 'Weekly'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show comparison count in tab header
    compare_count = len(st.session_state.get('compare_buildings', []))
    compare_label = f"Compare ({compare_count})" if compare_count > 0 else "Compare"

    tab1, tab2, tab3, tab4 = st.tabs(["Building Lookup", "Neighborhood Map", compare_label, "About"])

    with tab1:
        st.write("Search for any NYC building to see how it compares to its neighborhood and citywide.")

        # Time period selector
        time_period = st.radio(
            "Time period:",
            ["All time (2019+)", "Last 2 years", "Last 1 year"],
            horizontal=True,
            help="Filter violations by when they were filed"
        )

        if 'search_query_key' not in st.session_state:
            st.session_state['search_query_key'] = ""

        search_query = st.text_input("Enter address", placeholder="e.g., 123 Main Street, Brooklyn",
                                      key="search_query_key")
        
        if search_query:
            with st.spinner("Searching..."):
                results, suggestions = search_buildings(search_query)

            if results:
                display_building(results[0], time_period)
                if len(results) > 1:
                    st.write("---")
                    st.write("**Other matches:**")
                    for b in results[1:]:
                        addr = f"{b['housenumber']} {b['streetname']}, {BORO_MAP.get(str(b['boro']), '')}"
                        st.button(addr, key=f"result_{b['bbl']}", on_click=set_search, args=(addr,))
            elif suggestions:
                st.warning("No exact match. Did you mean:")
                for b in suggestions[:10]:
                    addr = f"{b['housenumber']} {b['streetname']}, {BORO_MAP.get(str(b['boro']), '')}"
                    st.button(addr, key=f"suggest_{b['bbl']}", on_click=set_search, args=(addr,))
            else:
                # Check if this address exists as a non-residential building
                nonres_building = check_nonresidential_buildings(search_query)
                if nonres_building:
                    bldgclass = nonres_building.get('bldgclass', '')
                    class_desc = get_building_class_description(bldgclass)
                    addr = nonres_building.get('pluto_address', 'This address')
                    boro = BORO_MAP.get(str(nonres_building.get('boro', '')), '')

                    st.info(f"""
                    **{addr}** ({boro}) is classified as a **{class_desc}** (Building Class: {bldgclass}).

                    This database only includes residential buildings (with 1+ residential units).
                    Non-residential buildings like offices, warehouses, schools, and places of worship are not included.
                    """)
                else:
                    st.info("No buildings found. Try a different address or check the spelling.")

        # Minimal data context - details in sidebar
        st.caption("Data from NYC Open Data (2019-present). Click sidebar for details.")
    
    with tab2:
        st.write("Compare neighborhoods by violation rates.")
        display_neighborhood_map()

    with tab3:
        st.write("Compare up to 3 buildings side-by-side.")
        display_comparison_view()

    with tab4:
        display_about_page()


if __name__ == "__main__":
    main()
