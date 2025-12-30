"""
Improve health data matching for asthma and lead data.

Issues addressed:
1. Asthma data uses NTA2010 names, violations use NTA2020 names
2. Lead data at UHF42 level (42 neighborhoods) - more granular than borough

This script:
1. Creates improved NTA2010 to NTA2020 crosswalk
2. Creates UHF42 to NTA2020 crosswalk
3. Re-joins health data with better matching
4. Recalculates the Housing Health Index

Input:
  - data/processed/nta_with_demographics.csv
  - data/health/childhood_asthma_*.csv
  - data/health/lead_poisoning_*.csv

Output: data/processed/nta_with_health_improved.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

from housing_health.paths import PROCESSED_DIR, HEALTH_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
)

# Manual NTA2010 to NTA2020 mappings for complex cases
# Format: NTA2020 name -> list of possible NTA2010 names
NTA_CROSSWALK = {
    # Bronx
    'Allerton': ['Allerton-Pelham Gardens'],
    'Bronxdale': ['Bronxdale'],
    'Claremont Village-Claremont (East)': ['Claremont-Bathgate'],
    'Co-op City': ['Co-op City'],
    'Concourse-Concourse Village': ['East Concourse-Concourse Village'],
    'Eastchester-Edenwald-Baychester': ['Eastchester-Edenwald-Baychester'],
    'Fordham Heights': ['Fordham South', 'Bedford Park-Fordham North'],
    'Highbridge': ['Highbridge'],
    'Hunts Point': ['Hunts Point-Mott Haven'],
    'Kingsbridge-Marble Hill': ['Kingsbridge Heights'],
    'Kingsbridge Heights-Van Cortlandt Village': ['Kingsbridge Heights', 'Van Cortlandt Village'],
    'Longwood': ['Longwood'],
    'Melrose': ['Melrose South-Mott Haven North'],
    'Morris Park': ['Morris Park-Bronxdale'],
    'Morrisania': ['Morrisania-Melrose'],
    'Mott Haven-Port Morris': ['Hunts Point-Mott Haven', 'Mott Haven-Port Morris'],
    'Mount Hope': ['Mount Hope'],
    'Norwood': ['Norwood'],
    'Parkchester': ['Parkchester'],
    'Pelham Bay-Country Club-City Island': ['Pelham Bay-Country Club-City Island'],
    'Pelham Parkway-Van Nest': ['Pelham Parkway'],
    'Soundview-Bruckner-Bronx River': ['Soundview-Castle Hill-Clason Point-Harding Park', 'West Farms-Bronx River'],
    'Soundview-Clason Point': ['Soundview-Castle Hill-Clason Point-Harding Park'],
    'Tremont': ['East Tremont', 'Tremont'],
    'University Heights (North)-Fordham': ['University Heights-Morris Heights', 'Fordham South'],
    'University Heights (South)-Morris Heights': ['University Heights-Morris Heights'],
    'West Farms': ['West Farms-Bronx River'],
    'Westchester Square': ['Westchester-Unionport'],
    'Williamsbridge-Olinville': ['Williamsbridge-Olinville'],

    # Brooklyn
    'Bedford-Stuyvesant (East)': ['Bedford', 'Stuyvesant Heights'],
    'Bedford-Stuyvesant (West)': ['Bedford', 'Stuyvesant Heights'],
    'Bensonhurst': ['Bensonhurst East', 'Bensonhurst West'],
    'Borough Park': ['Borough Park'],
    'Brighton Beach': ['Brighton Beach'],
    'Brownsville': ['Brownsville'],
    'Bushwick (East)': ['Bushwick North', 'Bushwick South'],
    'Bushwick (West)': ['Bushwick North', 'Bushwick South'],
    'Canarsie': ['Canarsie'],
    'Carroll Gardens-Cobble Hill-Gowanus-Red Hook': ['Carroll Gardens-Columbia Street-Red Hook', 'Cobble Hill'],
    'Clinton Hill': ['Clinton Hill'],
    'Crown Heights (North)': ['Crown Heights North'],
    'Crown Heights (South)': ['Crown Heights South'],
    'Cypress Hills': ['Cypress Hills-City Line'],
    'East Flatbush-Erasmus': ['Erasmus', 'East Flatbush-Farragut'],
    'East Flatbush-Farragut': ['East Flatbush-Farragut'],
    'East Flatbush-Remsen Village': ['Rugby-Remsen Village'],
    'East Flatbush-Rugby': ['Rugby-Remsen Village'],
    'East New York (North)': ['East New York', 'East New York (Pennsylvania Ave)'],
    'East New York-City Line': ['Cypress Hills-City Line', 'East New York'],
    'East New York-New Lots': ['East New York', 'New Lots'],
    'Flatbush': ['Flatbush'],
    'Flatbush (West)-Ditmas Park-Parkville': ['Flatbush', 'Kensington-Ocean Parkway'],
    'Flatlands': ['Flatlands'],
    'Fort Greene': ['Fort Greene'],
    'Gravesend (South)': ['Gravesend'],
    'Greenpoint': ['Greenpoint'],
    'Kensington': ['Kensington-Ocean Parkway'],
    'Midwood': ['Midwood'],
    'Ocean Hill': ['Ocean Hill'],
    'Park Slope': ['Park Slope-Gowanus'],
    'Prospect Heights': ['Prospect Heights'],
    'Prospect Lefferts Gardens-Wingate': ['Prospect Lefferts Gardens-Wingate'],
    'Sheepshead Bay-Manhattan Beach-Gerritsen Beach': ['Sheepshead Bay-Gerritsen Beach-Manhattan Beach'],
    'Spring Creek-Starrett City': ['Starrett City'],
    'Sunset Park (Central)': ['Sunset Park West', 'Sunset Park East'],
    'Sunset Park (East)': ['Sunset Park East'],
    'Williamsburg (East)': ['Williamsburg', 'East Williamsburg'],
    'Williamsburg (West)': ['Williamsburg', 'South Williamsburg'],
    'Windsor Terrace-South Slope': ['Windsor Terrace'],

    # Manhattan
    'Chelsea-Hudson Yards': ['Chelsea-Clinton'],
    'Chinatown-Two Bridges': ['Chinatown', 'Lower East Side'],
    'East Harlem (North)': ['East Harlem North', 'East Harlem South'],
    'East Harlem (South)': ['East Harlem South'],
    'East Midtown-Turtle Bay': ['Turtle Bay-East Midtown', 'Murray Hill-Kips Bay'],
    'East Village': ['East Village'],
    'Financial District-Battery Park City': ['Battery Park City-Lower Manhattan'],
    'Gramercy Park-Murray Hill': ['Murray Hill-Kips Bay', 'Gramercy'],
    'Hamilton Heights-Sugar Hill': ['Hamilton Heights'],
    'Harlem (Central)': ['Central Harlem North-Polo Grounds', 'Central Harlem South'],
    'Harlem (South)': ['Central Harlem South'],
    'Inwood': ['Washington Heights North', 'Marble Hill-Inwood'],
    'Lincoln Square': ['Lincoln Square'],
    'Lower East Side': ['Lower East Side'],
    'Manhattanville-West Harlem': ['Manhattanville'],
    'Midtown South-Flatiron-Union Square': ['Midtown-Midtown South', 'Hudson Yards-Chelsea-Flatiron-Union Square'],
    'Morningside Heights': ['Morningside Heights'],
    'SoHo-Little Italy-Hudson Square': ['SoHo-TriBeCa-Civic Center-Little Italy'],
    'Upper East Side (Central)': ['Upper East Side-Carnegie Hill', 'Yorkville'],
    'Upper East Side (North)': ['Upper East Side-Carnegie Hill'],
    'Upper West Side (Central)': ['Upper West Side'],
    'Upper West Side (North)': ['Upper West Side'],
    'Washington Heights (North)': ['Washington Heights North'],
    'Washington Heights (South)': ['Washington Heights South'],
    'West Village': ['West Village'],

    # Queens
    'Astoria (Central)': ['Astoria'],
    'Astoria (North)-Ditmars-Steinway': ['Astoria'],
    'Astoria (South)': ['Astoria', 'Old Astoria'],
    'Bayside': ['Bayside-Bayside Hills'],
    'Bellerose': ['Bellerose'],
    'Briarwood-Jamaica Hills': ['Briarwood-Jamaica Hills'],
    'Cambria Heights': ['Cambria Heights'],
    'College Point': ['College Point'],
    'Corona': ['Corona', 'North Corona'],
    'Douglaston-Little Neck': ['Douglaston-Little Neck'],
    'East Elmhurst': ['East Elmhurst'],
    'Elmhurst': ['Elmhurst'],
    'Far Rockaway-Bayswater': ['Far Rockaway-Bayswater'],
    'Flushing (Downtown)': ['Flushing'],
    'Flushing (Murray Hill)': ['Murray Hill'],
    'Flushing (West)-Pomonok': ['Flushing', 'Pomonok-Flushing Heights-Hillcrest'],
    'Forest Hills': ['Forest Hills'],
    'Fresh Meadows-Hillcrest': ['Fresh Meadows-Utopia', 'Pomonok-Flushing Heights-Hillcrest'],
    'Glen Oaks-Floral Park-New Hyde Park': ['Glen Oaks-Floral Park-New Hyde Park'],
    'Glendale': ['Glendale'],
    'Hollis': ['Hollis'],
    'Howard Beach-Lindenwood': ['Howard Beach'],
    'Jackson Heights (North)': ['Jackson Heights'],
    'Jackson Heights (South)': ['Jackson Heights'],
    'Jamaica': ['Jamaica'],
    'Jamaica Estates-Holliswood': ['Jamaica Estates-Holliswood'],
    'Kew Gardens': ['Kew Gardens'],
    'Kew Gardens Hills': ['Kew Gardens Hills'],
    'Laurelton': ['Laurelton'],
    'Long Island City (North)': ['Hunters Point-Sunnyside-West Maspeth'],
    'Long Island City (South)': ['Hunters Point-Sunnyside-West Maspeth'],
    'Maspeth': ['Maspeth'],
    'Middle Village': ['Middle Village'],
    'Oakland Gardens-Hollis Hills': ['Oakland Gardens'],
    'Ozone Park (North)': ['Ozone Park'],
    'Ozone Park (South)': ['Ozone Park'],
    'Queens Village': ['Queens Village'],
    'Rego Park': ['Rego Park'],
    'Richmond Hill': ['Richmond Hill'],
    'Ridgewood': ['Ridgewood'],
    'Rosedale': ['Rosedale'],
    'South Jamaica': ['South Jamaica'],
    'South Ozone Park': ['South Ozone Park'],
    'Springfield Gardens': ['Springfield Gardens North', 'Springfield Gardens South-Brookville'],
    'St. Albans': ["St. Albans"],
    'Sunnyside': ['Sunnyside', 'Hunters Point-Sunnyside-West Maspeth'],
    'Whitestone': ['Whitestone'],
    'Woodhaven': ['Woodhaven'],
    'Woodside': ['Woodside'],

    # Staten Island
    'Annadale-Huguenot-Prince\'s Bay-Woodrow': ['Annadale-Huguenot-Prince\'s Bay-Eltingville'],
    'Arden Heights-Rossville': ['Arden Heights'],
    'Charleston-Richmond Valley-Tottenville': ['Charleston-Richmond Valley-Tottenville'],
    'Eltingville-Annadale': ['Annadale-Huguenot-Prince\'s Bay-Eltingville'],
    'Grasmere-Arrochar-South Beach-Dongan Hills': ['Grasmere-Arrochar-Ft. Wadsworth', 'South Beach-Tottenville'],
    'Great Kills-Eltingville': ['Great Kills'],
    'Grymes Hill-Park Hill-Clifton': ['Grymes Hill-Clifton-Fox Hills'],
    'Mariner\'s Harbor-Arlington-Graniteville': ['Mariner\'s Harbor-Arlington-Port Ivory-Graniteville'],
    'New Brighton-Silver Lake-St. George': ['New Brighton-Silver Lake', 'St. George-New Brighton'],
    'New Dorp-Midland Beach-Grant City': ['New Dorp-Midland Beach'],
    'Port Richmond': ['Port Richmond'],
    'Rosebank-Shore Acres-Park Hill': ['Grymes Hill-Clifton-Fox Hills'],
    'Stapleton-Clifton-Fox Hills': ['Stapleton-Rosebank'],
    'Todt Hill-Emerson Hill-Lighthouse Hill-Manor Heights': ['Todt Hill-Emerson Hill-Heartland Village-Lighthouse Hill'],
    'Travis-Chelsea-Bloomfield-Bulls Head': ['New Springville-Bloomfield-Travis'],
    'West New Brighton-Snug Harbor-Livingston': ['West New Brighton-New Brighton-St. George'],
    'Westerleigh-Castleton Corners': ['Westerleigh'],
    'Willowbrook': ['Willowbrook'],
}

# UHF42 to NTA2020 mapping (UHF are larger than NTAs - one UHF maps to multiple NTAs)
UHF42_TO_NTAS = {
    'Hunts Point - Mott Haven': ['Hunts Point', 'Mott Haven-Port Morris', 'Longwood', 'Melrose'],
    'High Bridge - Morrisania': ['Highbridge', 'Morrisania', 'Concourse-Concourse Village', 'Mount Hope'],
    'Crotona -Tremont': ['Tremont', 'West Farms', 'Belmont', 'Crotona Park East'],
    'Fordham - Bronx Pk': ['Fordham Heights', 'Bedford Park', 'Norwood', 'University Heights (North)-Fordham',
                          'University Heights (South)-Morris Heights', 'Bronx Park'],
    'Pelham - Throgs Neck': ['Pelham Bay-Country Club-City Island', 'Pelham Parkway-Van Nest',
                             'Westchester Square', 'Castle Hill-Unionport', 'Throggs Neck-Schuylerville'],
    'Kingsbridge - Riverdale': ['Kingsbridge-Marble Hill', 'Kingsbridge Heights-Van Cortlandt Village',
                                'Riverdale-Fieldston-Spuyten Duyvil', 'North Riverdale-Fieldston-Riverdale'],
    'Northeast Bronx': ['Eastchester-Edenwald-Baychester', 'Williamsbridge-Olinville', 'Wakefield', 'Woodlawn'],
    'Southeast Bronx': ['Soundview-Bruckner-Bronx River', 'Soundview-Clason Point', 'Parkchester', 'Co-op City'],

    'Greenpoint': ['Greenpoint', 'Williamsburg (East)', 'Williamsburg (West)'],
    'Williamsburg - Bushwick': ['Bushwick (East)', 'Bushwick (West)', 'Williamsburg (East)', 'Williamsburg (West)'],
    'Downtown - Heights - Slope': ['Brooklyn Heights-Cobble Hill', 'DUMBO-Vinegar Hill-Downtown Brooklyn-Boerum Hill',
                                   'Park Slope', 'Carroll Gardens-Cobble Hill-Gowanus-Red Hook', 'Fort Greene'],
    'Bedford Stuyvesant - Crown Heights': ['Bedford-Stuyvesant (East)', 'Bedford-Stuyvesant (West)',
                                            'Crown Heights (North)', 'Crown Heights (South)', 'Clinton Hill'],
    'East New York': ['East New York (North)', 'East New York-City Line', 'East New York-New Lots', 'Cypress Hills'],
    'Sunset Park': ['Sunset Park (Central)', 'Sunset Park (East)', 'Windsor Terrace-South Slope'],
    'Borough Park': ['Borough Park', 'Kensington', 'Midwood'],
    'East Flatbush - Flatbush': ['Flatbush', 'East Flatbush-Erasmus', 'East Flatbush-Farragut',
                                  'East Flatbush-Remsen Village', 'East Flatbush-Rugby', 'Flatbush (West)-Ditmas Park-Parkville'],
    'Canarsie - Flatlands': ['Canarsie', 'Flatlands', 'Georgetown-Marine Park-Bergen Beach-Mill Basin'],
    'Bensonhurst - Bay Ridge': ['Bensonhurst', 'Bay Ridge', 'Dyker Heights', 'Bath Beach'],
    'Coney Island - Sheepshead Bay': ['Coney Island-Sea Gate', 'Brighton Beach', 'Sheepshead Bay-Manhattan Beach-Gerritsen Beach',
                                       'Gravesend (South)', 'Homecrest'],
    'Brownsville': ['Brownsville', 'Ocean Hill', 'Prospect Lefferts Gardens-Wingate'],

    'Washington Heights': ['Washington Heights (North)', 'Washington Heights (South)', 'Inwood'],
    'Central Harlem - Morningside Heights': ['Harlem (Central)', 'Harlem (South)', 'Manhattanville-West Harlem',
                                              'Hamilton Heights-Sugar Hill', 'Morningside Heights'],
    'East Harlem': ['East Harlem (North)', 'East Harlem (South)'],
    'Upper West Side': ['Upper West Side (Central)', 'Upper West Side (North)', 'Lincoln Square'],
    'Upper East Side': ['Upper East Side (Central)', 'Upper East Side (North)', 'Lenox Hill-Roosevelt Island'],
    'Chelsea - Clinton': ['Chelsea-Hudson Yards', 'Hell\'s Kitchen'],
    'Gramercy Park - Murray Hill': ['Gramercy Park-Murray Hill', 'East Midtown-Turtle Bay', 'Midtown South-Flatiron-Union Square'],
    'Greenwich Village - SoHo': ['West Village', 'East Village', 'SoHo-Little Italy-Hudson Square', 'NoHo-Greenwich Village'],
    'Union Square - Lower East Side': ['Lower East Side', 'Chinatown-Two Bridges'],
    'Lower Manhattan': ['Financial District-Battery Park City', 'Tribeca'],

    'Long Island City - Astoria': ['Astoria (Central)', 'Astoria (North)-Ditmars-Steinway', 'Astoria (South)',
                                    'Long Island City (North)', 'Long Island City (South)'],
    'West Queens': ['Sunnyside', 'Woodside', 'Maspeth', 'Elmhurst'],
    'Flushing - Clearview': ['Flushing (Downtown)', 'Flushing (Murray Hill)', 'Flushing (West)-Pomonok',
                             'Whitestone', 'College Point', 'Auburndale'],
    'Bayside - Little Neck': ['Bayside', 'Douglaston-Little Neck', 'Oakland Gardens-Hollis Hills', 'Glen Oaks-Floral Park-New Hyde Park'],
    'Ridgewood - Forest Hills': ['Ridgewood', 'Forest Hills', 'Glendale', 'Middle Village', 'Rego Park', 'Kew Gardens'],
    'Fresh Meadows': ['Fresh Meadows-Hillcrest', 'Kew Gardens Hills', 'Briarwood-Jamaica Hills', 'Jamaica Estates-Holliswood'],
    'Southwest Queens': ['Howard Beach-Lindenwood', 'Ozone Park (North)', 'Ozone Park (South)', 'Woodhaven', 'Richmond Hill'],
    'Jamaica': ['Jamaica', 'South Jamaica', 'Hollis', 'St. Albans'],
    'Southeast Queens': ['Queens Village', 'Bellerose', 'Cambria Heights', 'Laurelton', 'Rosedale', 'Springfield Gardens'],
    'Rockaways': ['Far Rockaway-Bayswater', 'Hammels-Arverne-Edgemere', 'Rockaway Beach-Breezy Point-Broad Channel'],

    'Port Richmond': ['Port Richmond', 'Mariner\'s Harbor-Arlington-Graniteville'],
    'Stapleton - St. George': ['Stapleton-Clifton-Fox Hills', 'New Brighton-Silver Lake-St. George',
                               'Rosebank-Shore Acres-Park Hill', 'Grymes Hill-Park Hill-Clifton'],
    'Willowbrook': ['Willowbrook', 'Westerleigh-Castleton Corners', 'West New Brighton-Snug Harbor-Livingston'],
    'South Beach - Tottenville': ['New Dorp-Midland Beach-Grant City', 'Great Kills-Eltingville', 'Grasmere-Arrochar-South Beach-Dongan Hills',
                                   'Annadale-Huguenot-Prince\'s Bay-Woodrow', 'Charleston-Richmond Valley-Tottenville',
                                   'Arden Heights-Rossville', 'Eltingville-Annadale', 'Todt Hill-Emerson Hill-Lighthouse Hill-Manor Heights',
                                   'Travis-Chelsea-Bloomfield-Bulls Head'],
}


def clean_name(name):
    """Normalize NTA name for matching."""
    if pd.isna(name):
        return ''
    name = str(name).lower().strip()
    # Remove parenthetical content for matching
    name = re.sub(r'\s*\([^)]*\)', '', name)
    # Remove common suffixes
    name = re.sub(r'\s+(north|south|east|west|central)$', '', name)
    return name


def fuzzy_score(s1, s2):
    """Calculate fuzzy match score."""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, clean_name(s1), clean_name(s2)).ratio()


def match_nta2010_to_nta2020(nta2020_name, asthma_ntas):
    """Find the best NTA2010 match for an NTA2020 name."""

    # First check manual crosswalk
    if nta2020_name in NTA_CROSSWALK:
        for nta2010_name in NTA_CROSSWALK[nta2020_name]:
            if nta2010_name in asthma_ntas:
                return nta2010_name, 1.0

    # Try exact match
    if nta2020_name in asthma_ntas:
        return nta2020_name, 1.0

    # Try fuzzy matching
    best_match = None
    best_score = 0

    for asthma_nta in asthma_ntas:
        score = fuzzy_score(nta2020_name, asthma_nta)
        if score > best_score:
            best_score = score
            best_match = asthma_nta

    if best_score >= 0.6:
        return best_match, best_score

    return None, 0


def get_uhf42_for_nta(nta_name, nta_borough):
    """Find the UHF42 neighborhood for an NTA."""

    for uhf_name, ntas in UHF42_TO_NTAS.items():
        if nta_name in ntas:
            return uhf_name

    # Try fuzzy matching
    for uhf_name, ntas in UHF42_TO_NTAS.items():
        for nta in ntas:
            if fuzzy_score(nta_name, nta) > 0.7:
                return uhf_name

    return None


def load_new_asthma_data(logger):
    """Load new asthma ED data from NYC EH Data Portal (2017-2019)."""
    from housing_health.paths import RAW_DIR

    # Look for new asthma files
    under4_file = RAW_DIR / "NYC EH Data Portal - Asthma emergency department visits (age 4 and under), by NTA (full table).csv"
    age5to17_file = RAW_DIR / "NYC EH Data Portal - Asthma emergency department visits (age 5 to 17), by NTA (full table).csv"

    if not under4_file.exists() or not age5to17_file.exists():
        logger.info("  New asthma files not found, falling back to old format")
        return None

    logger.info("Loading new asthma data (NYC EH Data Portal)...")

    def clean_num(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace(',', '').replace('*', '').strip()
        try:
            return float(s)
        except ValueError:
            return np.nan

    # Load under 4 data (only 2014-2016 available)
    df_under4 = pd.read_csv(under4_file)
    df_under4 = df_under4[df_under4['GeoType'] == 'NTA2010'].copy()
    df_under4['rate_under4'] = df_under4['Average annual rate per 10,000'].apply(clean_num)
    df_under4['count_under4'] = df_under4['Number'].apply(clean_num)
    logger.info(f"  Loaded under-4 data: {len(df_under4)} records")

    # Load 5-17 data - prefer 2017-2019
    df_5to17 = pd.read_csv(age5to17_file)
    df_5to17 = df_5to17[df_5to17['GeoType'] == 'NTA2010'].copy()

    # Get 2017-2019 data (more recent)
    df_5to17_recent = df_5to17[df_5to17['TimePeriod'] == '2017-2019'].copy()
    logger.info(f"  Loaded 5-17 data (2017-2019): {len(df_5to17_recent)} records")

    df_5to17_recent['rate_5to17'] = df_5to17_recent['Average annual rate per 10,000'].apply(clean_num)
    df_5to17_recent['count_5to17'] = df_5to17_recent['Average annual number'].apply(clean_num)
    df_5to17_recent = df_5to17_recent[['Geography', 'rate_5to17', 'count_5to17']]

    # Get under4 for 2014-2016
    df_under4_latest = df_under4[df_under4['TimePeriod'] == '2014-2016'][['Geography', 'rate_under4', 'count_under4']]

    # Combine
    combined = df_under4_latest.merge(df_5to17_recent, on='Geography', how='outer')

    # Calculate weighted average rate
    combined['total_count'] = combined['count_under4'].fillna(0) + combined['count_5to17'].fillna(0)
    combined['asthma_rate_per_10k'] = (
        (combined['rate_under4'].fillna(0) * combined['count_under4'].fillna(0) +
         combined['rate_5to17'].fillna(0) * combined['count_5to17'].fillna(0)) /
        combined['total_count'].replace(0, np.nan)
    )

    result = dict(zip(combined['Geography'], combined['asthma_rate_per_10k']))
    logger.info(f"  Combined childhood asthma for {len(result)} NTAs (2017-2019 for ages 5-17)")

    return result


def load_new_lead_data(logger):
    """Load new lead data by NTA from NYC EH Data Portal (2019)."""

    lead_nta_file = HEALTH_DIR / "NYC EH Data Portal - Elevated blood lead levels (under age 6), by NTA (full table).csv"

    if not lead_nta_file.exists():
        logger.info("  New lead NTA file not found, falling back to UHF42")
        return None

    logger.info(f"Loading new lead data by NTA (2019)...")
    df = pd.read_csv(lead_nta_file)

    # Filter to NTA level and 2019 (most recent)
    nta_lead = df[(df['GeoType'] == 'NTA2010') & (df['TimePeriod'] == '2019')].copy()
    logger.info(f"  Loaded {len(nta_lead)} NTA lead records for 2019")

    def clean_num(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace(',', '').replace('*', '').strip()
        try:
            return float(s)
        except ValueError:
            return np.nan

    # Rate column
    rate_col = "Rate (5+ mcg/dL) per 1,000 tested"
    nta_lead['lead_rate'] = nta_lead[rate_col].apply(clean_num)

    result = dict(zip(nta_lead['Geography'], nta_lead['lead_rate']))
    logger.info(f"  Lead rate range: {nta_lead['lead_rate'].min():.1f} - {nta_lead['lead_rate'].max():.1f} per 1k")

    return result


def improve_health_matching():
    """Main function to improve health data matching."""

    log_file = get_timestamped_log_filename("improve_health_matching")
    logger = setup_logger(__name__, log_file=log_file)

    log_step_start(logger, "Improve Health Data Matching")
    ensure_dirs_exist()

    # Load current NTA data
    input_file = PROCESSED_DIR / "nta_with_demographics.csv"
    logger.info(f"Loading NTA data from {input_file}")
    nta_df = pd.read_csv(input_file)
    logger.info(f"  Loaded {len(nta_df)} NTAs")

    # Try new asthma data first
    asthma_dict = load_new_asthma_data(logger)

    if asthma_dict is None:
        # Fall back to old format
        asthma_files = list(HEALTH_DIR.glob("childhood_asthma_*.csv"))
        if asthma_files:
            asthma_file = max(asthma_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading asthma data from {asthma_file}")
            asthma_df = pd.read_csv(asthma_file)

            asthma_nta = asthma_df[asthma_df['GeoType'].str.contains('NTA', case=False, na=False)]
            if 'TimePeriod' in asthma_nta.columns:
                most_recent = asthma_nta['TimePeriod'].max()
                asthma_nta = asthma_nta[asthma_nta['TimePeriod'] == most_recent]
                logger.info(f"  Using asthma time period: {most_recent}")

            asthma_dict = dict(zip(
                asthma_nta['Geography'],
                pd.to_numeric(asthma_nta['Average annual rate per 10,000'], errors='coerce')
            ))
            logger.info(f"  Loaded {len(asthma_dict)} asthma NTA records")

    if asthma_dict:
        asthma_ntas = set(asthma_dict.keys())
        logger.info("Matching asthma data to NTAs...")
        matched_count = 0
        for idx, row in nta_df.iterrows():
            nta_name = row['nta']
            match, score = match_nta2010_to_nta2020(nta_name, asthma_ntas)
            if match and match in asthma_dict:
                nta_df.loc[idx, 'asthma_rate_per_10k'] = asthma_dict[match]
                nta_df.loc[idx, 'asthma_match'] = match
                nta_df.loc[idx, 'asthma_match_score'] = score
                matched_count += 1

        logger.info(f"  Matched asthma data for {matched_count}/{len(nta_df)} NTAs")

    # Try new lead NTA data first
    lead_dict = load_new_lead_data(logger)

    if lead_dict:
        lead_ntas = set(lead_dict.keys())
        logger.info("Matching lead data (NTA level) to NTAs...")
        matched_count = 0
        for idx, row in nta_df.iterrows():
            nta_name = row['nta']
            match, score = match_nta2010_to_nta2020(nta_name, lead_ntas)
            if match and match in lead_dict:
                nta_df.loc[idx, 'lead_rate_per_1k'] = lead_dict[match]
                nta_df.loc[idx, 'lead_match'] = match
                nta_df.loc[idx, 'lead_match_score'] = score
                matched_count += 1

        logger.info(f"  Matched lead data for {matched_count}/{len(nta_df)} NTAs")

    else:
        # Fall back to UHF42 level
        lead_files = list(HEALTH_DIR.glob("lead_poisoning_*.csv"))
        if lead_files:
            lead_file = max(lead_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading lead data from {lead_file}")
            lead_df = pd.read_csv(lead_file)

            uhf_lead = lead_df[lead_df['geo_type'] == 'Neighborhood (UHF 42)']
            if 'time_period' in uhf_lead.columns:
                most_recent = uhf_lead['time_period'].max()
                uhf_lead = uhf_lead[uhf_lead['time_period'] == most_recent]
                logger.info(f"  Using lead time period: {most_recent}")

            rate_col = None
            for col in lead_df.columns:
                if 'Rate' in col and 'BLL>=5' in col and 'per 1,000' in col:
                    rate_col = col
                    break

            if rate_col:
                lead_dict = dict(zip(
                    uhf_lead['geo_area_name'],
                    pd.to_numeric(uhf_lead[rate_col], errors='coerce')
                ))
                logger.info(f"  Loaded {len(lead_dict)} UHF42 lead records")

                logger.info("Matching lead data to NTAs via UHF42...")
                matched_count = 0
                for idx, row in nta_df.iterrows():
                    nta_name = row['nta']
                    borough = row.get('borough', '')
                    uhf = get_uhf42_for_nta(nta_name, borough)
                    if uhf and uhf in lead_dict:
                        nta_df.loc[idx, 'lead_rate_per_1k'] = lead_dict[uhf]
                        nta_df.loc[idx, 'lead_uhf'] = uhf
                        matched_count += 1

                logger.info(f"  Matched lead data for {matched_count}/{len(nta_df)} NTAs")

    # Recalculate percentiles
    if 'asthma_rate_per_10k' in nta_df.columns:
        nta_df['asthma_pctl'] = nta_df['asthma_rate_per_10k'].rank(pct=True, na_option='keep') * 100
    if 'lead_rate_per_1k' in nta_df.columns:
        nta_df['lead_pctl'] = nta_df['lead_rate_per_1k'].rank(pct=True, na_option='keep') * 100

    # Save
    output_file = PROCESSED_DIR / "nta_with_health_improved.csv"
    logger.info(f"Saving to {output_file}")
    write_csv(nta_df, output_file)

    log_step_complete(logger, "Improve Health Data Matching")

    # Summary
    logger.info("=" * 60)
    logger.info("HEALTH DATA MATCHING SUMMARY")
    logger.info("=" * 60)

    asthma_matched = nta_df['asthma_rate_per_10k'].notna().sum()
    lead_matched = nta_df['lead_rate_per_1k'].notna().sum()

    logger.info(f"Asthma data matched: {asthma_matched}/{len(nta_df)} NTAs ({asthma_matched/len(nta_df)*100:.1f}%)")
    logger.info(f"Lead data matched: {lead_matched}/{len(nta_df)} NTAs ({lead_matched/len(nta_df)*100:.1f}%)")

    if 'asthma_rate_per_10k' in nta_df.columns:
        logger.info(f"\nAsthma rate range: {nta_df['asthma_rate_per_10k'].min():.1f} - {nta_df['asthma_rate_per_10k'].max():.1f} per 10k")
    if 'lead_rate_per_1k' in nta_df.columns:
        logger.info(f"Lead rate range: {nta_df['lead_rate_per_1k'].min():.1f} - {nta_df['lead_rate_per_1k'].max():.1f} per 1k")

    return nta_df


if __name__ == "__main__":
    df = improve_health_matching()

    asthma_matched = df['asthma_rate_per_10k'].notna().sum()
    lead_matched = df['lead_rate_per_1k'].notna().sum()

    print(f"\n✅ Improved health data matching!")
    print(f"  Asthma matched: {asthma_matched}/{len(df)} NTAs")
    print(f"  Lead matched: {lead_matched}/{len(df)} NTAs")
