import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.parse
import time
import pickle
import requests
from bs4 import BeautifulSoup
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CSVS_PATH
from helpers.file_manager import FileManager
from helpers.property_modifier import PropertyModifier
from unidecode import unidecode
import csv
sys.stdout.reconfigure(encoding='utf-8')

class ArtistUrl:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        
class ArtFactsUrls:   
    def lowercase(self, value):
        value = str(value)
        value = value.lower()
        return value 

    def delete_unnecessary_sign(self, value):
        value = str(value)
        if '(' in value:
            return value.split(' (')[0]
        if '&' in value:
            return value.split(' &')[0]
        return value

    def remove_spaces(self, value):
        value = str(value)
        if "." in value:
            value = value.replace("."," ")
        return value   

    def encode_text(self, unique_artists):
        unique_artists_encoded =[]
        for one in unique_artists:
            one = unidecode(one)
            unique_artists_encoded.append(one)
        return unique_artists_encoded
            
    def get_urls(self, artists, base_url):
        array =[]
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(options=options)
        for artist in artists:
            driver.get(base_url)
            wait = WebDriverWait(driver, 10)
            try:
                search_bar = wait.until(EC.presence_of_element_located((By.NAME, "q")))
                search_bar.clear()
                search_bar.send_keys(artist)

                search_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "app-js-components-search-SearchField__input")))  # Update with correct class
                search_button.click()
                time.sleep(5) 
                result_items = wait.until(EC.visibility_of_all_elements_located((By.XPATH, "//div[@class='app-js-components-search-SearchResult__container']//ol//li[@class='app-js-components-search-SearchResult__resultItem']//a")))

                print(f"Found {len(result_items)} result items.")

                if result_items:
                    href = result_items[0].get_attribute("href")
                    print(href)
                    one = ArtistUrl(artist, href)
                    array.append(one)
                else:
                    print("No search results found.")
            except Exception as e:
                print(e)
        return array

    def write_to_csv(self, filename, artists):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            writer.writerow(["Name", "Url"])

            for artist in artists:
                writer.writerow([artist.name, artist.url])    
        
    def main_artfacts_url(self, CSV_OUTPUT, gather_information):
        if gather_information:
            print("Urls from ArtFacts will be collected...")
            data_to_enrich = FileManager.read_all_csvs(CSVS_PATH)            
            df = PropertyModifier.get_unique_dataframe(data_to_enrich, "Artist name")  
                        
            df["Artist name"] = df["Artist name"].apply(self.lowercase)
            df["Artist name"] = df["Artist name"].apply(self.delete_unnecessary_sign)
            
            df["Name"] = df["Artist name"]
            df["Name"] = df["Name"].apply(self.remove_spaces) 

            artists = df["Name"]
            unique_artists_encoded = self.encode_text(artists)
                              
            BASE_URL = "https://artfacts.net/"

            array = self.get_urls(unique_artists_encoded, BASE_URL)
            self.write_to_csv(CSV_OUTPUT, array)
        else:
            print("Urls from ArtFacts will NOT be collected")

