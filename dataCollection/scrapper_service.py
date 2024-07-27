import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode for no UI
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--log-level=3")  # Suppress unnecessary log messages

# Initialize the WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL of the ICC cricket matches page
url = "https://www.icc-cricket.com/matches"

# List to hold the scraped data
data = []

try:
    # Navigate to the URL
    driver.get(url)

    # Wait for the match block elements to load
    WebDriverWait(driver, 300).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'match-block'))
    )

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find the section containing match details
    matches_section = soup.find_all('div', class_='match-block')

    for match in matches_section:
        try:
            # Extract team names
            teams = match.find_all('p', class_='team-names')
            if len(teams) < 2:
                print("Skipping a match block due to insufficient team names.")
                continue
            team1 = teams[0].text.strip()
            team2 = teams[1].text.strip()

            # Extract match format
            match_format = match.find('span', class_='match-format').text.strip()

            # Extract venue
            venue = match.find('p', class_='match-venue').text.strip()

            # Extract date and time
            date_time_str = match.find('time').text.strip()
            try:
                date_time_obj = datetime.strptime(date_time_str, '%d %B %Y, %H:%M %p')
            except ValueError as ve:
                print(f"Error parsing date for match {team1} vs {team2}: {ve}")
                continue

            # Extract flags
            flags = match.find_all('span', class_='team-flag')
            if len(flags) < 2:
                print("Skipping a match block due to insufficient flags.")
                continue
            flag1 = flags[0].find('img')['src']
            flag2 = flags[1].find('img')['src']

            # Append the extracted data to the list
            data.append([team1, team2, match_format, venue, date_time_obj, flag1, flag2])

        except Exception as e:
            print(f"Error extracting data for a match block: {e}")

except Exception as e:
    print(f"Error scraping the ICC cricket site: {e}")

finally:
    # Close the WebDriver
    driver.quit()

# Save the scraped data to a CSV file
csv_file = "cricket_matches.csv"
try:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Team 1', 'Team 2', 'Match Format', 'Venue', 'Date & Time', 'Flag 1', 'Flag 2'])
        writer.writerows(data)
    print(f"Data successfully saved to {csv_file}")
except Exception as e:
    print(f"Error saving data to CSV file: {e}")
