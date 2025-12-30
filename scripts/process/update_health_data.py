"""
Update health data with new 2017-2019 asthma data.

This script:
1. Loads new asthma ED visit data for ages 0-4 and 5-17
2. Combines them into total childhood asthma rates
3. Uses the more recent 2017-2019 data where available
4. Applies NTA2010 to NTA2020 crosswalk for name matching
5. Keeps existing lead data (UHF42 to NTA mapping)
6. Updates the housing health index with fresh data

Input:
  - data/raw/NYC EH Data Portal - Asthma emergency department visits (age 4 and under), by NTA (full table).csv
  - data/raw/NYC EH Data Portal - Asthma emergency department visits (age 5 to 17), by NTA (full table).csv
  - data/health/lead_poisoning_2025-12-26.csv

Output: data/processed/nta_with_health_updated.csv
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np

from housing_health.paths import PROCESSED_DIR, RAW_DIR, HEALTH_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
    log_dataframe_info,
)


# NTA2010 to NTA2020 crosswalk (comprehensive mapping)
NTA2010_TO_NTA2020 = {
    # Bronx
    'Claremont-Bathgate': 'Claremont-Bathgate',
    'Eastchester-Edenwald-Baychester': 'Eastchester-Edenwald-Baychester',
    'Bedford Park-Fordham North': 'Bedford Park-Fordham North',
    'Belmont': 'Belmont',
    'Bronxdale': 'Bronxdale',
    'West Farms-Bronx River': 'West Farms-Bronx River',
    'Soundview-Castle Hill-Clason Point-Harding Park': 'Soundview-Castle Hill-Clason Point-Harding Park',
    'Pelham Bay-Country Club-City Island': 'Pelham Bay-Country Club-City Island',
    'Co-op City': 'Co-Op City',  # Note: hyphen difference
    'East Concourse-Concourse Village': 'Concourse-Concourse Village',
    'East Tremont': 'East Tremont',
    'North Riverdale-Fieldston-Riverdale': 'North Riverdale-Fieldston-Riverdale',
    'Highbridge': 'Highbridge',
    'Hunts Point': 'Hunts Point',
    'Van Cortlandt Village': 'Van Cortlandt Village',
    'Spuyten Duyvil-Kingsbridge': 'Spuyten Duyvil-Kingsbridge',
    'Kingsbridge Heights': 'Kingsbridge Heights',
    'Allerton-Pelham Gardens': 'Allerton-Pelham Gardens',
    'Longwood': 'Longwood',
    'Melrose South-Mott Haven North': 'Melrose South-Mott Haven North',
    'Morrisania-Melrose': 'Morrisania-Melrose',
    'University Heights-Morris Heights': 'University Heights-Morris Heights',
    'Van Nest-Morris Park-Westchester Square': 'Van Nest-Morris Park-Westchester Square',
    'Mott Haven-Port Morris': 'Mott Haven-Port Morris',
    'Fordham South': 'Fordham South',
    'Mount Hope': 'Mount Hope',
    'Norwood': 'Norwood',
    'Williamsbridge-Olinville': 'Williamsbridge-Olinville',
    'Parkchester': 'Parkchester',
    'Pelham Parkway': 'Pelham Parkway',
    'Schuylerville-Throgs Neck-Edgewater Park': 'Schuylerville-Throgs Neck-Edgewater Park',
    'Soundview-Bruckner': 'Soundview-Bruckner',
    'Westchester-Unionport': 'Westchester-Unionport',
    'Woodlawn-Wakefield': 'Woodlawn-Wakefield',
    'West Concourse': 'West Concourse',
    'Crotona Park East': 'Crotona Park East',

    # Brooklyn
    'Brooklyn Heights-Cobble Hill': 'Brooklyn Heights-Cobble Hill',
    'Sheepshead Bay-Gerritsen Beach-Manhattan Beach': 'Sheepshead Bay-Gerritsen Beach-Manhattan Beach',
    'Brighton Beach': 'Brighton Beach',
    'Seagate-Coney Island': 'Seagate-Coney Island',
    'West Brighton': 'West Brighton',
    'Homecrest': 'Homecrest',
    'Gravesend': 'Gravesend',
    'Bath Beach': 'Bath Beach',
    'Bensonhurst West': 'Bensonhurst West',
    'Bensonhurst East': 'Bensonhurst East',
    'Dyker Heights': 'Dyker Heights',
    'Bay Ridge': 'Bay Ridge',
    'Sunset Park West': 'Sunset Park West',
    'Carroll Gardens-Columbia Street-Red Hook': 'Carroll Gardens-Cobble Hill-Gowanus-Red Hook',
    'Sunset Park East': 'Sunset Park East',
    'Stuyvesant Heights': 'Stuyvesant Heights',
    'Park Slope-Gowanus': 'Park Slope-Gowanus',
    'DUMBO-Vinegar Hill-Downtown Brooklyn-Boerum Hill': 'DUMBO-Vinegar Hill-Downtown Brooklyn-Boerum Hill',
    'Windsor Terrace': 'South Slope-Windsor Terrace',
    'Kensington-Ocean Parkway': 'Kensington-Ocean Parkway',
    'Flatbush': 'Flatbush',
    'Midwood': 'Midwood',
    'Madison': 'Madison',
    'Georgetown-Marine Park-Bergen Beach-Mill Basin': 'Georgetown-Marine Park-Bergen Beach-Mill Basin',
    'Ocean Parkway South': 'Ocean Parkway South',
    'Canarsie': 'Canarsie',
    'Flatlands': 'Flatlands',
    'Prospect Lefferts Gardens-Wingate': 'Prospect Lefferts Gardens-Wingate',
    'Crown Heights North': 'Crown Heights North',
    'Crown Heights South': 'Crown Heights South',
    'Prospect Heights': 'Prospect Heights',
    'Fort Greene': 'Fort Greene',
    'Clinton Hill': 'Clinton Hill',
    'Williamsburg': 'South Williamsburg',
    'North Side-South Side': 'Williamsburg',
    'Bedford': 'Bedford',
    'Greenpoint': 'Greenpoint',
    'Bushwick North': 'Bushwick North',
    'Bushwick South': 'Bushwick South',
    'Ocean Hill': 'Ocean Hill',
    'Brownsville': 'Brownsville',
    'East New York': 'East New York',
    'Cypress Hills-City Line': 'Cypress Hills-City Line',
    'East New York (Pennsylvania Ave)': 'East New York (Pennsylvania Ave)',
    'Borough Park': 'Borough Park',
    'East Williamsburg': 'East Williamsburg',
    'East Flatbush-Farragut': 'East Flatbush-Farragut',
    'Starrett City': 'Starrett City',
    'Erasmus': 'Erasmus',
    'Rugby-Remsen Village': 'Rugby-Remsen Village',

    # Manhattan
    'Marble Hill-Inwood': 'Marble Hill-Inwood',
    'Central Harlem North-Polo Grounds': 'Central Harlem North-Polo Grounds',
    'Hamilton Heights': 'Hamilton Heights',
    'Manhattanville': 'Manhattanville',
    'Morningside Heights': 'Morningside Heights',
    'Central Harlem South': 'Central Harlem South',
    'Upper West Side': 'Upper West Side',
    'Hudson Yards-Chelsea-Flatiron-Union Square': 'Hudson Yards-Chelsea-Flat Iron-Union Square',
    'Lincoln Square': 'Lincoln Square',
    'Clinton': 'Clinton',
    'Midtown-Midtown South': 'Midtown-Midtown South',
    'Turtle Bay-East Midtown': 'Turtle Bay-East Midtown',
    'Murray Hill-Kips Bay': 'Murray Hill-Kips Bay',
    'Gramercy': 'Gramercy',
    'East Village': 'East Village',
    'West Village': 'West Village',
    'SoHo-TriBeCa-Civic Center-Little Italy': 'SoHo-TriBeCa-Civic Center-Little Italy',
    'Battery Park City-Lower Manhattan': 'Battery Park City-Lower Manhattan',
    'Chinatown': 'Chinatown',
    'Lower East Side': 'Lower East Side',
    'Lenox Hill-Roosevelt Island': 'Lenox Hill-Roosevelt Island',
    'Yorkville': 'Yorkville',
    'East Harlem South': 'East Harlem South',
    'East Harlem North': 'East Harlem North',
    'Washington Heights North': 'Washington Heights North',
    'Washington Heights South': 'Washington Heights South',
    'Upper East Side-Carnegie Hill': 'Upper East Side-Carnegie Hill',
    'Stuyvesant Town-Cooper Village': 'Stuyvesant Town-Cooper Village',

    # Queens
    'South Jamaica': 'South Jamaica',
    'Springfield Gardens North': 'Springfield Gardens North',
    'Springfield Gardens South-Brookville': 'Springfield Gardens South-Brookville',
    'Rosedale': 'Rosedale',
    'Jamaica Estates-Holliswood': 'Jamaica Estates-Holliswood',
    'Hollis': 'Hollis',
    'St. Albans': 'St. Albans',
    'Breezy Point-Belle Harbor-Rockaway Park-Broad Channel': 'Breezy Point-Belle Harbor-Rockaway Park-Broad Channel',
    'Hammels-Arverne-Edgemere': 'Hammels-Arverne-Edgemere',
    'Far Rockaway-Bayswater': 'Far Rockaway-Bayswater',
    'Forest Hills': 'Forest Hills',
    'Rego Park': 'Rego Park',
    'Glendale': 'Glendale',
    'Ridgewood': 'Ridgewood',
    'Middle Village': 'Middle Village',
    'Flushing': 'Flushing',
    'College Point': 'College Point',
    'Corona': 'Corona',
    'North Corona': 'North Corona',
    'East Elmhurst': 'East Elmhurst',
    'Jackson Heights': 'Jackson Heights',
    'Elmhurst': 'Elmhurst',
    'Maspeth': 'Maspeth',
    'Hunters Point-Sunnyside-West Maspeth': 'Hunters Point-Sunnyside-West Maspeth',
    'Cambria Heights': 'Cambria Heights',
    'Queens Village': 'Queens Village',
    'Briarwood-Jamaica Hills': 'Briarwood-Jamaica Hills',
    'Kew Gardens Hills': 'Kew Gardens Hills',
    'Pomonok-Flushing Heights-Hillcrest': 'Pomonok-Flushing Heights-Hillcrest',
    'Fresh Meadows-Utopia': 'Fresh Meadows-Utopia',
    'Oakland Gardens': 'Oakland Gardens',
    'Bellerose': 'Bellerose',
    'Glen Oaks-Floral Park-New Hyde Park': 'Glen Oaks-Floral Park-New Hyde Park',
    'Douglas Manor-Douglaston-Little Neck': 'Douglas Manor-Douglaston-Little Neck',
    'Bayside-Bayside Hills': 'Bayside-Bayside Hills',
    'Ft. Totten-Bay Terrace-Clearview': 'Ft. Totten-Bay Terrace-Clearview',
    'Auburndale': 'Auburndale',
    'Whitestone': 'Whitestone',
    'Elmhurst-Maspeth': 'Elmhurst-Maspeth',
    'Murray Hill': 'Murray Hill',
    'East Flushing': 'East Flushing',
    'Woodhaven': 'Woodhaven',
    'Richmond Hill': 'Richmond Hill',
    'South Ozone Park': 'South Ozone Park',
    'Ozone Park': 'Ozone Park',
    'Lindenwood-Howard Beach': 'Lindenwood-Howard Beach',
    'Kew Gardens': 'Kew Gardens',
    'Jamaica': 'Jamaica',
    'Queensboro Hill': 'Queensboro Hill',
    'Woodside': 'Woodside',
    'Laurelton': 'Laurelton',
    'Queensbridge-Ravenswood-Long Island City': 'Queensbridge-Ravenswood-Long Island City',
    'Astoria': 'Astoria',
    'Old Astoria': 'Old Astoria',
    'Steinway': 'Steinway',
    'Baisley Park': 'Baisley Park',

    # Staten Island
    'Annadale-Huguenot-Prince\'s Bay-Eltingville': 'Annadale-Huguenot-Prince\'s Bay-Eltingville',
    'New Springville-Bloomfield-Travis': 'New Springville-Bloomfield-Travis',
    'Westerleigh': 'Westerleigh',
    'Grymes Hill-Clifton-Fox Hills': 'Grymes Hill-Clifton-Fox Hills',
    'Charleston-Richmond Valley-Tottenville': 'Charleston-Richmond Valley-Tottenville',
    'Mariner\'s Harbor-Arlington-Port Ivory-Graniteville': 'Mariner\'s Harbor-Arlington-Port Ivory-Graniteville',
    'Grasmere-Arrochar-Ft. Wadsworth': 'Grasmere-Arrochar-Ft. Wadsworth',
    'West New Brighton-New Brighton-St. George': 'West New Brighton-New Brighton-St. George',
    'Todt Hill-Emerson Hill-Heartland Village-Lighthouse Hill': 'Todt Hill-Emerson Hill-Heartland Village-Lighthouse Hill',
    'Oakwood-Oakwood Beach': 'Oakwood-Oakwood Beach',
    'Port Richmond': 'Port Richmond',
    'Rossville-Woodrow': 'Rossville-Woodrow',
    'New Brighton-Silver Lake': 'New Brighton-Silver Lake',
    'Old Town-Dongan Hills-South Beach': 'Old Town-Dongan Hills-South Beach',
    'Stapleton-Rosebank': 'Stapleton-Rosebank',
    'New Dorp-Midland Beach': 'New Dorp-Midland Beach',
    'Arden Heights': 'Arden Heights',
    'Great Kills': 'Great Kills',
}

# UHF42 to NTA2020 mapping (for lead data)
UHF42_TO_NTA = {
    # Bronx
    '101': ['Kingsbridge Heights', 'Bedford Park-Fordham North', 'Fordham South', 'University Heights-Morris Heights', 'Mount Hope'],
    '102': ['Eastchester-Edenwald-Baychester', 'Williamsbridge-Olinville', 'Woodlawn-Wakefield'],
    '103': ['Highbridge', 'West Concourse', 'Concourse-Concourse Village'],
    '104': ['Pelham Bay-Country Club-City Island', 'Pelham Parkway', 'Schuylerville-Throgs Neck-Edgewater Park', 'Westchester-Unionport', 'Parkchester'],
    '105': ['Crotona Park East', 'East Tremont', 'Belmont', 'Bronxdale', 'Van Nest-Morris Park-Westchester Square'],
    '106': ['Norwood', 'Allerton-Pelham Gardens', 'Van Cortlandt Village', 'Spuyten Duyvil-Kingsbridge', 'North Riverdale-Fieldston-Riverdale'],
    '107': ['Hunts Point', 'Longwood', 'Melrose South-Mott Haven North', 'Mott Haven-Port Morris', 'Morrisania-Melrose'],
    '108': ['Soundview-Castle Hill-Clason Point-Harding Park', 'Soundview-Bruckner', 'Claremont-Bathgate', 'West Farms-Bronx River'],
    '109': ['Co-Op City'],

    # Brooklyn
    '201': ['Williamsburg', 'South Williamsburg', 'Greenpoint', 'East Williamsburg'],
    '202': ['Brooklyn Heights-Cobble Hill', 'DUMBO-Vinegar Hill-Downtown Brooklyn-Boerum Hill', 'Fort Greene', 'Clinton Hill', 'Carroll Gardens-Cobble Hill-Gowanus-Red Hook', 'Prospect Heights', 'Park Slope-Gowanus'],
    '203': ['Bedford', 'Crown Heights North', 'Crown Heights South', 'Stuyvesant Heights'],
    '204': ['Brownsville', 'Ocean Hill', 'East New York', 'East New York (Pennsylvania Ave)', 'Cypress Hills-City Line', 'Starrett City'],
    '205': ['Sunset Park West', 'Sunset Park East', 'South Slope-Windsor Terrace'],
    '206': ['Borough Park', 'Kensington-Ocean Parkway', 'Ocean Parkway South', 'Midwood'],
    '207': ['Flatbush', 'East Flatbush-Farragut', 'Erasmus', 'Prospect Lefferts Gardens-Wingate', 'Rugby-Remsen Village'],
    '208': ['Canarsie', 'Flatlands', 'Georgetown-Marine Park-Bergen Beach-Mill Basin'],
    '209': ['Bay Ridge', 'Dyker Heights', 'Bath Beach', 'Gravesend', 'Bensonhurst West', 'Bensonhurst East'],
    '210': ['Brighton Beach', 'Sheepshead Bay-Gerritsen Beach-Manhattan Beach', 'Homecrest', 'Madison', 'West Brighton', 'Seagate-Coney Island'],
    '211': ['Bushwick North', 'Bushwick South'],

    # Manhattan
    '301': ['Washington Heights North', 'Washington Heights South', 'Marble Hill-Inwood'],
    '302': ['Central Harlem North-Polo Grounds', 'Central Harlem South', 'Hamilton Heights', 'Manhattanville', 'Morningside Heights'],
    '303': ['East Harlem North', 'East Harlem South'],
    '304': ['Upper West Side', 'Lincoln Square'],
    '305': ['Upper East Side-Carnegie Hill', 'Lenox Hill-Roosevelt Island', 'Yorkville'],
    '306': ['Stuyvesant Town-Cooper Village', 'Murray Hill-Kips Bay', 'Gramercy', 'Turtle Bay-East Midtown'],
    '307': ['Clinton', 'Midtown-Midtown South', 'Hudson Yards-Chelsea-Flat Iron-Union Square'],
    '308': ['West Village', 'SoHo-TriBeCa-Civic Center-Little Italy', 'Battery Park City-Lower Manhattan'],
    '309': ['Lower East Side', 'East Village', 'Chinatown'],

    # Queens
    '401': ['Long Island City', 'Queensbridge-Ravenswood-Long Island City', 'Hunters Point-Sunnyside-West Maspeth', 'Woodside'],
    '402': ['Astoria', 'Old Astoria', 'Steinway'],
    '403': ['Flushing', 'Auburndale', 'Whitestone', 'College Point', 'Ft. Totten-Bay Terrace-Clearview', 'Bayside-Bayside Hills'],
    '404': ['Douglas Manor-Douglaston-Little Neck', 'Glen Oaks-Floral Park-New Hyde Park', 'Bellerose'],
    '405': ['Ridgewood', 'Glendale', 'Middle Village', 'Maspeth'],
    '406': ['Fresh Meadows-Utopia', 'Oakland Gardens', 'Queensboro Hill', 'Kew Gardens Hills', 'Pomonok-Flushing Heights-Hillcrest', 'East Flushing'],
    '407': ['Elmhurst', 'Elmhurst-Maspeth', 'Jackson Heights', 'Corona', 'North Corona', 'East Elmhurst'],
    '408': ['Jamaica', 'Jamaica Estates-Holliswood', 'Briarwood-Jamaica Hills', 'Kew Gardens', 'Forest Hills', 'Rego Park'],
    '409': ['Queens Village', 'Cambria Heights', 'Hollis', 'St. Albans', 'Laurelton', 'Rosedale', 'Springfield Gardens North', 'Springfield Gardens South-Brookville'],
    '410': ['Far Rockaway-Bayswater', 'Hammels-Arverne-Edgemere', 'Breezy Point-Belle Harbor-Rockaway Park-Broad Channel'],
    '411': ['South Jamaica', 'Baisley Park', 'South Ozone Park', 'Ozone Park', 'Richmond Hill', 'Woodhaven', 'Lindenwood-Howard Beach'],
    '412': ['Murray Hill'],

    # Staten Island
    '501': ['Port Richmond', 'Mariner\'s Harbor-Arlington-Port Ivory-Graniteville', 'West New Brighton-New Brighton-St. George', 'New Brighton-Silver Lake'],
    '502': ['Westerleigh', 'Grymes Hill-Clifton-Fox Hills', 'Stapleton-Rosebank', 'Grasmere-Arrochar-Ft. Wadsworth'],
    '503': ['New Springville-Bloomfield-Travis', 'Todt Hill-Emerson Hill-Heartland Village-Lighthouse Hill', 'Old Town-Dongan Hills-South Beach'],
    '504': ['Annadale-Huguenot-Prince\'s Bay-Eltingville', 'Charleston-Richmond Valley-Tottenville', 'Rossville-Woodrow', 'Great Kills', 'Arden Heights', 'Oakwood-Oakwood Beach', 'New Dorp-Midland Beach'],
}


def clean_number(x):
    """Clean numeric values from CSV (handles commas, asterisks)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(',', '').replace('*', '').strip()
    if s == '' or s == '-':
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_asthma_data(logger):
    """Load and combine asthma data for ages 0-4 and 5-17."""

    # Find asthma files
    under4_file = RAW_DIR / "NYC EH Data Portal - Asthma emergency department visits (age 4 and under), by NTA (full table).csv"
    age5to17_file = RAW_DIR / "NYC EH Data Portal - Asthma emergency department visits (age 5 to 17), by NTA (full table).csv"

    if not under4_file.exists():
        logger.warning(f"Under 4 asthma file not found: {under4_file}")
        return None
    if not age5to17_file.exists():
        logger.warning(f"5-17 asthma file not found: {age5to17_file}")
        return None

    logger.info("Loading asthma data files...")

    # Load under 4 data (only 2014-2016 available)
    df_under4 = pd.read_csv(under4_file)
    df_under4 = df_under4[df_under4['GeoType'] == 'NTA2010'].copy()
    df_under4['rate_under4'] = df_under4['Average annual rate per 10,000'].apply(clean_number)
    df_under4['count_under4'] = df_under4['Number'].apply(clean_number)
    df_under4 = df_under4[['Geography', 'TimePeriod', 'rate_under4', 'count_under4']]
    logger.info(f"  Loaded under-4 data: {len(df_under4)} NTA records")

    # Load 5-17 data (has 2014-2016 AND 2017-2019)
    df_5to17 = pd.read_csv(age5to17_file)
    df_5to17 = df_5to17[df_5to17['GeoType'] == 'NTA2010'].copy()

    # Prefer 2017-2019 data (more recent)
    df_5to17_recent = df_5to17[df_5to17['TimePeriod'] == '2017-2019'].copy()
    df_5to17_old = df_5to17[df_5to17['TimePeriod'] == '2014-2016'].copy()

    logger.info(f"  5-17 data: {len(df_5to17_recent)} NTAs with 2017-2019, {len(df_5to17_old)} NTAs with 2014-2016")

    # Use 2017-2019 where available, fallback to 2014-2016
    df_5to17_recent['rate_5to17'] = df_5to17_recent['Average annual rate per 10,000'].apply(clean_number)
    df_5to17_recent['count_5to17'] = df_5to17_recent['Average annual number'].apply(clean_number)
    df_5to17_recent = df_5to17_recent[['Geography', 'rate_5to17', 'count_5to17']]

    # Get under4 for 2014-2016 to match with 2017-2019 age 5-17
    # (best available approximation)
    df_under4_latest = df_under4[df_under4['TimePeriod'] == '2014-2016'][['Geography', 'rate_under4', 'count_under4']]

    # Combine datasets
    combined = df_under4_latest.merge(df_5to17_recent, on='Geography', how='outer')

    # Calculate total childhood rate (weighted average by count)
    combined['total_count'] = combined['count_under4'].fillna(0) + combined['count_5to17'].fillna(0)
    combined['asthma_rate_per_10k'] = (
        (combined['rate_under4'].fillna(0) * combined['count_under4'].fillna(0) +
         combined['rate_5to17'].fillna(0) * combined['count_5to17'].fillna(0)) /
        combined['total_count'].replace(0, np.nan)
    )

    # Apply NTA2010 to NTA2020 crosswalk
    combined['nta'] = combined['Geography'].map(
        lambda x: NTA2010_TO_NTA2020.get(x, x)
    )

    logger.info(f"  Combined childhood asthma data for {len(combined)} NTAs")
    logger.info(f"  Rate range: {combined['asthma_rate_per_10k'].min():.1f} - {combined['asthma_rate_per_10k'].max():.1f} per 10k")
    logger.info(f"  Note: Using 2017-2019 data for ages 5-17 (most recent available)")

    return combined[['nta', 'asthma_rate_per_10k', 'total_count']].rename(
        columns={'total_count': 'asthma_count'}
    )


def load_lead_data(logger):
    """Load lead data from UHF42 and expand to NTA level."""

    lead_file = HEALTH_DIR / "lead_poisoning_2025-12-26.csv"
    if not lead_file.exists():
        # Try old file
        lead_files = list(HEALTH_DIR.glob("lead_*.csv"))
        if not lead_files:
            logger.warning("No lead data file found")
            return None
        lead_file = max(lead_files, key=lambda x: x.stat().st_mtime)

    logger.info(f"Loading lead data from {lead_file}")

    df = pd.read_csv(lead_file)

    # Get most recent UHF42 data (2016)
    uhf_data = df[
        (df['geo_type'] == 'Neighborhood (UHF 42)') &
        (df['time_period'] == 2016)
    ].copy()

    # Extract UHF code from geo_area_id
    uhf_data['uhf_code'] = uhf_data['geo_area_id'].astype(str)

    # Get lead rate column
    rate_col = [c for c in uhf_data.columns if 'Rate  BLL>=5' in c and 'per 1,000' in c][0]
    uhf_data['lead_rate_per_1k'] = pd.to_numeric(uhf_data[rate_col], errors='coerce')

    logger.info(f"  Loaded {len(uhf_data)} UHF42 neighborhoods with lead data")

    # Expand to NTA level using mapping
    nta_lead_records = []
    for uhf_code, ntas in UHF42_TO_NTA.items():
        uhf_row = uhf_data[uhf_data['uhf_code'] == uhf_code]
        if len(uhf_row) > 0:
            rate = uhf_row['lead_rate_per_1k'].values[0]
            for nta in ntas:
                nta_lead_records.append({
                    'nta': nta,
                    'lead_rate_per_1k': rate,
                    'uhf_source': uhf_code
                })

    nta_lead = pd.DataFrame(nta_lead_records)
    logger.info(f"  Expanded to {len(nta_lead)} NTA records using UHF42 mapping")

    return nta_lead


def update_health_data():
    """Main function to update health data with new sources.

    Strategy: Keep existing health data (which has good coverage) but update
    asthma rates with more recent 2017-2019 data where we can match.
    """

    # Setup logging
    log_file = get_timestamped_log_filename("update_health_data")
    logger = setup_logger(__name__, log_file=log_file)

    log_step_start(logger, "Update Health Data")

    # Ensure directories exist
    ensure_dirs_exist()

    # Load improved NTA data (which has best existing health matching)
    input_file = PROCESSED_DIR / "nta_with_health_improved.csv"
    if not input_file.exists():
        input_file = PROCESSED_DIR / "nta_with_demographics.csv"
    if not input_file.exists():
        input_file = PROCESSED_DIR / "nta_with_health.csv"

    logger.info(f"Loading NTA data from {input_file}")
    nta_df = pd.read_csv(input_file)
    log_dataframe_info(logger, nta_df, "NTA data")

    # Store original asthma data
    original_asthma = nta_df['asthma_rate_per_10k'].copy() if 'asthma_rate_per_10k' in nta_df.columns else None
    original_coverage = nta_df['asthma_rate_per_10k'].notna().sum() if original_asthma is not None else 0

    logger.info(f"Original asthma coverage: {original_coverage}/{len(nta_df)} NTAs")
    logger.info(f"Original asthma data vintage: 2014-2016")

    # Load new asthma data
    asthma_df = load_asthma_data(logger)

    if asthma_df is not None:
        # Create lookup dict from new asthma data
        asthma_lookup = dict(zip(asthma_df['nta'], asthma_df['asthma_rate_per_10k']))

        # Count how many we can update
        updated_count = 0
        new_matches = 0

        for idx, row in nta_df.iterrows():
            nta_name = row['nta']
            new_rate = asthma_lookup.get(nta_name)

            if new_rate is not None and not pd.isna(new_rate):
                if pd.isna(nta_df.loc[idx, 'asthma_rate_per_10k']):
                    new_matches += 1
                else:
                    updated_count += 1
                nta_df.loc[idx, 'asthma_rate_per_10k'] = new_rate

        final_coverage = nta_df['asthma_rate_per_10k'].notna().sum()
        logger.info(f"\nAsthma data update results:")
        logger.info(f"  Updated with 2017-2019 data: {updated_count} NTAs")
        logger.info(f"  New matches: {new_matches} NTAs")
        logger.info(f"  Final coverage: {final_coverage}/{len(nta_df)} NTAs ({final_coverage/len(nta_df)*100:.1f}%)")

    # Recalculate asthma percentile
    if 'asthma_rate_per_10k' in nta_df.columns:
        nta_df['asthma_pctl'] = nta_df['asthma_rate_per_10k'].rank(pct=True, na_option='keep') * 100
        logger.info(f"  Recalculated asthma percentiles")

    # Save updated data
    output_file = PROCESSED_DIR / "nta_with_health_updated.csv"
    logger.info(f"\nSaving to {output_file}...")
    write_csv(nta_df, output_file)

    log_step_complete(logger, "Update Health Data")

    # Summary
    logger.info("=" * 60)
    logger.info("HEALTH DATA UPDATE SUMMARY")
    logger.info("=" * 60)

    if 'asthma_rate_per_10k' in nta_df.columns:
        logger.info("\nAsthma Data (2017-2019 for ages 5-17):")
        logger.info(f"  Coverage: {nta_df['asthma_rate_per_10k'].notna().sum()}/{len(nta_df)} NTAs")
        logger.info(f"  Mean rate: {nta_df['asthma_rate_per_10k'].mean():.1f} per 10k")
        logger.info(f"  Max rate: {nta_df['asthma_rate_per_10k'].max():.1f} per 10k")

        # Top 5 highest asthma
        top5 = nta_df.nlargest(5, 'asthma_rate_per_10k')[['nta', 'borough', 'asthma_rate_per_10k']]
        logger.info("\nTop 5 Highest Asthma Rates:")
        for _, row in top5.iterrows():
            logger.info(f"  {row['nta']}: {row['asthma_rate_per_10k']:.1f} per 10k")

    if 'lead_rate_per_1k' in nta_df.columns:
        logger.info("\nLead Data (2016, UHF42->NTA):")
        logger.info(f"  Coverage: {nta_df['lead_rate_per_1k'].notna().sum()}/{len(nta_df)} NTAs")
        logger.info(f"  Mean rate: {nta_df['lead_rate_per_1k'].mean():.1f} per 1k")

    return nta_df


if __name__ == "__main__":
    df = update_health_data()
    print(f"\n Updated health data for {len(df)} neighborhoods!")

    if 'asthma_rate_per_10k' in df.columns:
        print(f"\n Asthma coverage: {df['asthma_rate_per_10k'].notna().sum()}/{len(df)} NTAs")
        print(f"  Using 2017-2019 data (3 years more recent!)")

    if 'lead_rate_per_1k' in df.columns:
        print(f"\n Lead coverage: {df['lead_rate_per_1k'].notna().sum()}/{len(df)} NTAs")
