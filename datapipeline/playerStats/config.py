import os

# Configuration settings
BASE_DIR = "hdfs://192.168.245.142:8020/usr/ravi/t20"
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', '1_rawData')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', '2_processedData')

# Country codes mapping
# Ensure all country codes are included
COUNTRY_CODES = {
            'LES': 'Lesotho', 'BUL': 'Bulgaria', 'VAN': 'Vanuatu', 'ROM': 'Romania', 'Aut': 'Austria', 'COK': 'Cook Islands', 'Fran': 'France', 'SRB': 'Serbia', 'PAK': 'Pakistan', 'HUN': 'Hungary', 'CYP': 'Cyprus', 'Fiji': 'Fiji', 'FIN': 'Finland', 'EST': 'Estonia', 'CHN': 'China', 'GRC': 'Greece', 'CAM': 'Cambodia', 'GUE': 'Guernsey', 'SEY': 'Seychelles', 'JPN': 'Japan', 'TAN': 'Tanzania', 'JER': 'Jersey', 'QAT': 'Qatar', 'ENG': 'England', 'UGA': 'Uganda', 'BER': 'Bermuda', 'CZK-R': 'Czech Republic', 'CAY': 'Cayman Islands', 'IRE': 'Ireland', 'Mali': 'Mali', 'BRA': 'Brazil', 'SUI': 'Switzerland', 'Peru': 'Peru', 'Mex': 'Mexico', 'MOZ': 'Mozambique', 'Samoa': 'Samoa', 'HKG': 'Hong Kong', 'BAN': 'Bangladesh', 'SL': 'Sri Lanka', 'PNG': 'Papua New Guinea', 'ZIM': 'Zimbabwe', 'GHA': 'Ghana', 'SWZ': 'Eswatini', # Swaziland's official name now is Eswatini
            'MYAN': 'Myanmar', 'IND': 'India', 'USA': 'United States of America', 'NEP': 'Nepal', 'AFG': 'Afghanistan', 'PAN': 'Panama', 'NGA': 'Nigeria', 'SLE': 'Sierra Leone', 'ESP': 'Spain', 'Bhm': 'Bahamas', 'TKY': 'Turkey', 'MWI': 'Malawi', 'WI': 'West Indies', 'IOM': 'Isle of Man', 'THA': 'Thailand', 'SWA': 'Eswatini',
            'SKOR': 'South Korea', 'GMB': 'Gambia', 'ISR': 'Israel', 'KUW': 'Kuwait', 'Belg': 'Belgium', 'GER': 'Germany', 'ITA': 'Italy', 'CAN': 'Canada', 'MDV': 'Maldives', 'Blz': 'Belize', 'DEN': 'Denmark', 'INA': 'Indonesia', 'KENYA': 'Kenya', 'LUX': 'Luxembourg', 'STHEL': 'Saint Helena', 'BHR': 'Bahrain', 'KSA': 'Saudi Arabia', 'MLT': 'Malta', 'Arg': 'Argentina', 'MNG': 'Mongolia', 'AUS': 'Australia', 'GIBR': 'Gibraltar', 'SGP': 'Singapore', 'Chile': 'Chile', 'UAE': 'United Arab Emirates', 'NZ': 'New Zealand', 'SCOT': 'Scotland', 'BHU': 'Bhutan', 'MAS': 'Malaysia', 'BOT': 'Botswana', 'CRC': 'Costa Rica', 'PHI': 'Philippines', 'NAM': 'Namibia', 'RWN': 'Rwanda', 'OMA': 'Oman', 'NOR': 'Norway', 'CRT': 'Croatia', 'SWE': 'Sweden', 'Iran': 'Iran', 'PORT': 'Portugal', 'NED': 'Netherlands', 'SA': 'South Africa', 'SVN': 'Slovenia', 'GUE': 'Guernsey', 'MDV': 'Maldives', 'BHM': 'Bahamas', 'SWE': 'Sweden', 'MLT': 'Malta', 'ITA': 'Italy',
        }
