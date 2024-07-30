from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from bs4 import BeautifulSoup
import time

# Set up Selenium WebDriver
options = Options()
options.add_argument('--headless')  # Run in headless mode (without opening the browser)
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

# Base URL
base_url = 'https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;page={1};template=results;type=batting'

# Function to parse table from a single page
import pandas as pd
all_tables = pd.read_html(base_url)
print(all_tables)