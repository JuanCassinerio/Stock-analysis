'''
pip install selenium
pip install webdriver_manager
'''

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

import os
import time

options=Options()
options.add_experimental_option("detach",True)
driver=webdriver.Chrome(service=Service(ChromeDriverManager().install(),options=options))

driver.get("https://www.rava.com/perfil/DOLAR%20CCL") #page to scrape, more specific = less steps = smaller code


 
button = driver.find_element(By.XPATH, "//button[@data-v-19e8c281='']") #find element (source code of webpage)
driver.execute_script("arguments[0].click();", button) #execute button to download file



# file is being downloaded to downloads folder on PC and open
download_dir = r"C:\Users\Usuario\Downloads"
previous_files = set(os.listdir(download_dir))

current_files = set(os.listdir(download_dir))
new_files = current_files - previous_files



# Function to check for new files in the download directory
def find_new_file(download_dir):
    previous_files = set(os.listdir(download_dir))

    while True:
        time.sleep(10)  # Adjust sleep time as needed
        current_files = set(os.listdir(download_dir))
        new_files = current_files - previous_files
        print("hola")
        if new_files:
            return new_files.pop()  # Return the first new file found
        previous_files = current_files



# After download completes (replace with your clicking logic)
downloaded_file = find_new_file(download_dir)
print(f"Downloaded file: {downloaded_file}")