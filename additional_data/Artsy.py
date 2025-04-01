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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unidecode import unidecode
import csv
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException
from config import CSVS_PATH
from helpers.file_manager import FileManager
from helpers.property_modifier import PropertyModifier
sys.stdout.reconfigure(encoding='utf-8')

class ArtistArtsy:
    def __init__(self, name, nationality, years, followers, description, fact_header_1, fact_header_2, fact_header_3, fact_header_4, fact_info_1, fact_info_2, fact_info_3, fact_info_4):
        self.name = name
        self.nationality = nationality
        self.years = years
        self.followers = followers
        self.description = description
        self.fact_header_1 = fact_header_1
        self.fact_header_2 = fact_header_2
        self.fact_header_3 = fact_header_3
        self.fact_header_4 = fact_header_4
        self.fact_info_1 = fact_info_1
        self.fact_info_2 = fact_info_2
        self.fact_info_3 = fact_info_3
        self.fact_info_4 = fact_info_4        
    def __str__(self):
        return (f"Artist: {self.name}; Nationality: {self.nationality}; Years: {self.years}; Followers: {self.followers}; Description: {self.description}; {self.fact_header_1}: {self.fact_info_1}; {self.fact_header_2}: {self.fact_info_2}; {self.fact_header_3}: {self.fact_info_3}; {self.fact_header_4}: {self.fact_info_4}")

class Artsy:               
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
        if " " in value:
            value = value.replace(" ","-") 
        if "." in value:
            value = value.replace(".","")
        return value   

    def encode_text(self, unique_artists):
        unique_artists_encoded =[]
        for one in unique_artists:
            one = unidecode(one)
            unique_artists_encoded.append(one)
        return unique_artists_encoded

    def form_urls(self, base_url, all_artists):
        urls = []
        for i in range(len(all_artists)):
            artist = all_artists[i]
            formed_url = base_url+artist
            urls.append(formed_url)
        return urls
            
          
    def get_status_of_request(self, urls):
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(options=options)
        artists_artsy = []
        print()
        for i in range(len(urls)):
            url = urls[i]
            r = requests.get(url)
            status = r.status_code
            print(status)
            time.sleep(5)  
            if status == 200:
                try:
                    artist_artsy = self.get_artist_information(driver, url)             
                    artists_artsy.append(artist_artsy)
                except Exception as e:
                    print(f"Error in {url}: {e}")  
        driver.quit()
        return artists_artsy

    def write_to_csv(self, filename, artists):
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Nationality", "Years", "Followers", "Description",
                            "Fact Header 1", "Fact Info 1", "Fact Header 2", "Fact Info 2", 
                            "Fact Header 3", "Fact Info 3", "Fact Header 4", "Fact Info 4"])

            for artist in artists:
                writer.writerow([artist.name, artist.nationality, artist.years, artist.followers, artist.description, 
                                artist.fact_header_1, artist.fact_info_1, artist.fact_header_2, artist.fact_info_2, 
                                artist.fact_header_3, artist.fact_info_3, artist.fact_header_4, artist.fact_info_4])

    def get_artist_information(self, driver, url): 
        driver.get(url)
        time.sleep(3)      
        try:
            try:
                artist_name_element = driver.find_element(By.XPATH, "//div[@class='Box-sc-15se88d-0 fmwhTL']//h1")
                artist_name_text = artist_name_element.text.strip()
            except Exception as e:
                artist_name_text=""
            try:
                artist_nationality_element = driver.find_element(By.XPATH, "//div[@class='Box-sc-15se88d-0 fmwhTL']//h2")
                artist_nationality_text = artist_nationality_element.text.strip()
                if "," in artist_nationality_text:
                    nationality, years = artist_nationality_text.split(",")
                else:
                    nationality = artist_nationality_text
                    years = ""
                    # birth_years, death_years = years.split("-")
            except Exception as e:
                nationality = ""
                years = ""           
            try:
                followers_element = driver.find_element(By.XPATH, "//div[@class='Box-sc-15se88d-0 fmwhTL']//div[@class='Box-sc-15se88d-0 Flex-sc-cw39ct-0 ecvDKa']//div[@class='Box-sc-15se88d-0 Text-sc-18gcpao-0 cZekcQ gviZDz']")
                followers_text = followers_element.text.strip().split(" ")[0]
                if "k" in followers_text:
                    followers_text = followers_text.replace("k","000")     
            except Exception as e:
                followers_text=""  
            
            try:
                
                button_descrription = driver.find_element(By.XPATH, "//button[@class='Clickable-sc-10cr82y-0 iPyMNF']")
                button_descrription.click()
                description_element =  driver.find_element(By.XPATH, "//div[@class='Box-sc-15se88d-0 Text-sc-18gcpao-0 HTML__Container-sc-1im40xc-0 gpKROX eaQyqh ArtistHeader__Bio-sc-f1ae9cbf-0 jcwOPs']//div[@class='ReadMore__Container-sc-1bqy0ya-0 bcRuMR']//div[@class='Box-sc-15se88d-0']//p")
                description = description_element.text.strip()
            except Exception as e:
                description=""    
            
            try:
                try:
                    facts_header_elements = driver.find_elements(By.XPATH, "//div[@class='Box-sc-15se88d-0 Flex-sc-cw39ct-0 gqXHBT']")
                    facts_header = [fact.text.strip() for fact in facts_header_elements if fact.text.strip()]
                    print(facts_header)
                    try:
                        overlay = driver.find_element(By.XPATH, "//div[contains(@class, 'Box-sc-15se88d-0 Text-sc-18gcpao-0 dglQCu oxDIp')]")
                        driver.execute_script("arguments[0].style.display = 'none';", overlay)
                        print("Overlay hidden successfully")
                        time.sleep(1) 
                    except:
                        print("No overlay detected")

                    buttons = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, "//button[@class='Clickable-sc-10cr82y-0 fbnHxf']")))
                    
                    buttons_amount = len(facts_header)
                    print(buttons_amount)
                    k = 0
                    if buttons_amount == 0:
                        print("No buttons found! Check the XPath.")
                    else:
                        for button in buttons:
                            try:
                                WebDriverWait(driver, 5).until(EC.element_to_be_clickable(button))
                                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                                time.sleep(1)  
                                
                                try:
                                    driver.execute_script("arguments[0].click();", button)
                                    print("JavaScript Click Successful")
                                    k+=1
                                    if buttons_amount == k:
                                        break
                                except ElementClickInterceptedException:
                                    print("JS click failed, using ActionChains")
                                    actions = ActionChains(driver)
                                    actions.move_to_element(button).click().perform()                            
                                time.sleep(1)
                            except TimeoutException:
                                print("Timeout: Button not clickable after 5 seconds.")

                    facts_info_elements = driver.find_elements(By.XPATH, "//div[@class='Box-sc-15se88d-0 Text-sc-18gcpao-0 ArtistCareerHighlight__Description-sc-96a31884-0 dOEQPP fYwUsr ciXvbe']")
                    print(facts_info_elements)
                    facts_info = [fact.text.strip() for fact in facts_info_elements if fact.text.strip()]
                    print("\n".join(facts_info)) 
                    print(len(facts_info))
                    for i in range(len(facts_info)):
                        print(f"{facts_header[i]} - {facts_info[i]}")       
            
                except:
                    facts_header = ""
                    facts_info = ""
                
            except Exception as e:
                facts_header = ""
                facts_info = ""
                print(e)

            
        except Exception as e:
            print("Error:", e)
            
        fact_header_1 = fact_header_2 = fact_header_3 = fact_header_4 = fact_info_1 = fact_info_2 = fact_info_3 = fact_info_4 = ""  
        for i in range(len(facts_info)):
            print(i)
            if i==0:
                fact_header_1 = facts_header[i]
                fact_info_1 = facts_info[i]            
            if i==1:
                fact_header_2 = facts_header[i]
                fact_info_2 = facts_info[i]      
            if i==2:
                fact_header_3 = facts_header[i]
                fact_info_3 = facts_info[i]      
            if i==3:
                fact_header_4 = facts_header[i]
                fact_info_4 = facts_info[i]      
        artist_artsy = ArtistArtsy(artist_name_text, nationality, years, followers_text, description, fact_header_1, fact_header_2, fact_header_3, fact_header_4, fact_info_1, fact_info_2, fact_info_3, fact_info_4)
        print(artist_artsy)
            
        return artist_artsy
        
    def main_artsy(self, CSV_FILENAME_OUTPUT, gather_information):
        if gather_information:
            print("Information from Artsy will be collected...")
            data_to_enrich = FileManager.read_all_csvs(CSVS_PATH)            
            df = PropertyModifier.get_unique_dataframe(data_to_enrich, "Artist name")            
            df["Name"] = df["Artist name"]
            df["Name"] = df["Name"].apply(self.remove_spaces) 

            artists = df["Name"]
            unique_artists = artists.unique()
            unique_artists_encoded = self.encode_text(unique_artists)            

            BASE_URL = "https://www.artsy.net/artist/"
            urls = self.form_urls(BASE_URL, unique_artists_encoded)
            
            artists_artsy = self.get_status_of_request(urls)
            self.write_to_csv(CSV_FILENAME_OUTPUT, artists_artsy)
        else:
            print("Information from Artsy will NOT be collected")

