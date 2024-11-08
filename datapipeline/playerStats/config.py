import os

# Configuration settings
BASE_DIR = "hdfs://192.168.245.142:8020/usr/ravi/t20"
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', '1_rawData')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', '2_processedData')

# Country codes mapping
COUNTRY_CODES = {
    'LES': 'Lesotho', 'BUL': 'Bulgaria', 'VAN': 'Vanuatu', 'ROM': 'Romania', 'Aut': 'Austria',
    'COK': 'Cook Islands', 'Fran': 'France', 'SRB': 'Serbia', 'PAK': 'Pakistan', 'HUN': 'Hungary',
    # ... (rest of the country codes)
    'MLT': 'Malta', 'ITA': 'Italy',
}
