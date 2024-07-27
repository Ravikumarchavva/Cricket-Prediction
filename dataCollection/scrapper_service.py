from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

# Set up Selenium
browser = webdriver.Chrome()

try:
    browser.get('https://www.icc-cricket.com/fixtures-results')
    
    # Wait for the page to load and the "All Formats" dropdown to be clickable
    WebDriverWait(browser, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, '.si-selected-title'))
    )
    
    # Click the "All Formats" dropdown
    allFormats = browser.find_element(By.CSS_SELECTOR, '.si-selected-title')
    allFormats.click()
    
    # Wait for the dropdown options to be visible
    WebDriverWait(browser, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".si-select-list"))
    )
    
    # Click the third button (T20)
    t20_button = browser.find_elements(By.CSS_SELECTOR, ".si-list-item")[2]  # Index 2 for the third button
    t20_button.click()
    
    # Wait for the page to load the T20 matches
    time.sleep(3)  # Adjust sleep time if necessary
    
    # Get the updated page source and parse with BeautifulSoup
    html_content = browser.page_source
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all match containers
    matches = soup.find_all(class_='si-tab-container-item')
    if not matches:
        print("No match containers found.")
    
    for match in matches:
        # Find all titles within the current match container
        titles = match.find_all(class_='si-title')
        if not titles:
            print("No titles found in this match container.")
            
        for title in titles:
            print(title.get_text(strip=True))  # Print the text content of each title
        print('\n')  # Print a newline for separation between matches

finally:
    browser.quit()
