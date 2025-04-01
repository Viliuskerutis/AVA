import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import urllib.parse
import time
import pickle
import requests
from bs4 import BeautifulSoup
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unidecode import unidecode
import csv
sys.stdout.reconfigure(encoding='utf-8')
from ArtistUrl import ArtistUrl
  
class ArtFactsArtist:
    def __init__(self, name, birth_year, birth_city, birth_country, death_year, death_city, death_country, gender, nationality, movement, media, period, ranking,
                 verified_exhibitions, solo_exhibition, group_exhibitions, biennials, art_fairs,
                 exhibitions_top_country_1, exhibitions_top_country_2, exhibitions_top_country_3,
                 exhibitions_top_country_1_no, exhibitions_top_country_2_no, exhibitions_top_country_3_no,
                 exhibitions_most_at_1, exhibitions_most_at_2, exhibitions_most_at_3,
                 exhibitions_most_at_1_no, exhibitions_most_at_2_no, exhibitions_most_at_3_no,
                 exhibitions_most_with_1,exhibitions_most_with_2,exhibitions_most_with_3,
                 exhibitions_most_with_rank_1, exhibitions_most_with_rank_2,exhibitions_most_with_rank_3,
                 exhibitions_most_with_1_no,exhibitions_most_with_2_no,exhibitions_most_with_3_no,
                 notable_exhibitions_at_1, notable_exhibitions_at_2, notable_exhibitions_at_3, description):
        self.name = name
        self.birth_year = birth_year
        self.birth_city = birth_city
        self.birth_country = birth_country
        self.death_year = death_year
        self.death_city = death_city
        self.death_country = death_country
        self.gender = gender
        self.nationality = nationality
        self.movement = movement
        self.media = media
        self.period = period
        self.ranking = ranking
        self.verified_exhibitions = verified_exhibitions
        self.solo_exhibitions = solo_exhibition
        self.group_exhibitions = group_exhibitions
        self.biennials = biennials
        self.art_fairs = art_fairs
        self.exhibitions_top_country_1 = exhibitions_top_country_1
        self.exhibitions_top_country_2 = exhibitions_top_country_2
        self.exhibitions_top_country_3 = exhibitions_top_country_3 
        self.exhibitions_top_country_1_no = exhibitions_top_country_1_no,
        self.exhibitions_top_country_2_no = exhibitions_top_country_2_no,
        self.exhibitions_top_country_3_no = exhibitions_top_country_3_no,
        self.exhibitions_most_at_1 = exhibitions_most_at_1
        self.exhibitions_most_at_2 = exhibitions_most_at_2
        self.exhibitions_most_at_3 = exhibitions_most_at_3 
        self.exhibitions_most_at_1_no = exhibitions_most_at_1_no,
        self.exhibitions_most_at_2_no = exhibitions_most_at_2_no,
        self.exhibitions_most_at_3_no = exhibitions_most_at_3_no,
        self.exhibitions_most_with_1=exhibitions_most_with_1,
        self.exhibitions_most_with_2=exhibitions_most_with_2,
        self.exhibitions_most_with_3=exhibitions_most_with_3,
        self.exhibitions_most_with_rank_1=exhibitions_most_with_rank_1, 
        self.exhibitions_most_with_rank_2=exhibitions_most_with_rank_2, 
        self.exhibitions_most_with_rank_3=exhibitions_most_with_rank_3,
        self.exhibitions_most_with_1_no=exhibitions_most_with_1_no,
        self.exhibitions_most_with_2_no=exhibitions_most_with_2_no, 
        self.exhibitions_most_with_3_no=exhibitions_most_with_3_no,
        self.notable_exhibitions_at_1 = notable_exhibitions_at_1
        self.notable_exhibitions_at_2 = notable_exhibitions_at_2
        self.notable_exhibitions_at_3 = notable_exhibitions_at_3 
        self.description = description
        
    
    def __str__(self):
        return (f"Artist: {self.name};\nBirth year: {self.birth_year};\nBirth City: {self.birth_city}; \n"
                f"Birth Country: {self.birth_country};\nDeath year: {self.death_year};\n"
                f"Death City: {self.death_city}; \nDeath Country: {self.death_country};\nGender: {self.gender};\n"
                f"Nationality: {self.nationality}; \nMovement: {self.movement};\nMedia: {self.media};\n"
                f"Period: {self.period}; \nRanking: {self.ranking};\nVerified exhibitions: {self.verified_exhibitions};\n"
                f"Solo exhibitions: {self.solo_exhibitions}; \nGroup exhibitions: {self.group_exhibitions};\nBiennals: {self.biennials};\n"
                f"Art fairs: {self.art_fairs};\n"
                f"Exhibitions top country 1: {self.exhibitions_top_country_1};\nExhibitions top country 2: {self.exhibitions_top_country_2};\nExhibitions top country 3: {self.exhibitions_top_country_3};\n"
                f"Exhibitions top country 1 No: {self.exhibitions_top_country_1_no};\nExhibitions top country 2 No: {self.exhibitions_top_country_2_no};\nExhibitions top country 3 No: {self.exhibitions_top_country_3_no};\n"
                f"Exhibitions most at 1: {self.exhibitions_most_at_1};\nExhibitions most at 2: {self.exhibitions_most_at_2};\nExhibitions most at 3: {self.exhibitions_most_at_3};\n"
                f"Exhibitions most at 1 No: {self.exhibitions_most_at_1_no};\nExhibitions most at 2 No: {self.exhibitions_most_at_2_no};\nExhibitions most at 3 No: {self.exhibitions_most_at_3_no};\n"
                f"Exhibitions most with 1: {self.exhibitions_most_with_1};\nExhibitions most with 2: {self.exhibitions_most_with_2};\nExhibitions most with 3: {self.exhibitions_most_with_3};\n"
                f"Exhibitions most with rank 1: {self.exhibitions_most_with_rank_1};\nExhibitions most with rank 2: {self.exhibitions_most_with_rank_2};\nExhibitions most with rank 3: {self.exhibitions_most_with_rank_3};\n"
                f"Exhibitions most with 1 No: {self.exhibitions_most_with_1_no};\nExhibitions most with 2 No: {self.exhibitions_most_with_2_no};\nExhibitions most with 3 No: {self.exhibitions_most_with_3_no};\n"                 
                f"Notable Exhibitions At 1: {self.notable_exhibitions_at_1};\n"
                f"Notable Exhibitions At 2: {self.notable_exhibitions_at_2};\nNotable Exhibitions At 3: {self.notable_exhibitions_at_3};\nDescription: {self.description}"
                )

class ArtFacts:
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

    def encode_text(self, list):
        unique_artists_encoded =[]
        for one in unique_artists:
            one = unidecode(one)
            unique_artists_encoded.append(one)
        return unique_artists_encoded

    def get_artist_information(self, driver, url, CSV_FILENAME_OUTPUT): 
        driver.get(url)
        text_to_find="("

        time.sleep(3)      
        try:
            try:
                artist_name_element  = driver.find_element(By.CLASS_NAME, 'app-js-components-PageTitle-PageTitle__pageTitle')
                artist_name = artist_name_element.find_element(By.TAG_NAME, 'h1').text.strip()
            except Exception as e:
                artist_name=""
                
            try:     
                birth_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-Spotlight-Spotlight__summaryDetail']//table[@class='app-js-components-Spotlight-Spotlight__info']//tr[td[contains(text(), 'Born')]]/td[2]")
                birth_text = birth_element.text.strip()
                if "|" in birth_text:
                    birth_year, birth_city_text = birth_text.split(" | ")
                    if text_to_find in birth_text:
                        birth_city = birth_city_text.split(" (")[0].strip()
                        birth_country = birth_city_text.split(" (")[1].strip(")")
                    else:
                        birth_city=birth_city_text.text.strip()
                        birth_country=""
                else:
                    birth_year=birth_text
                    birth_city=""
                    birth_country=""
            except Exception as e:
                birth_year=""  
                birth_city=""
                birth_country=""
            
            try:
                death_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-Spotlight-Spotlight__summaryDetail']//table[@class='app-js-components-Spotlight-Spotlight__info']//tr[td[contains(text(), 'Death')]]/td[2]")
                death_text = death_element.text.strip()           
                
                if "|" in death_text:
                    death_year, death_city_text = death_text.split(" | ")
                    if text_to_find in death_text:
                        death_city = death_city_text.split(" (")[0].strip()
                        death_country = death_city_text.split(" (")[1].strip(")")
                    else:
                        death_city=death_city_text.text.strip()
                        death_country=""
                else:
                    death_year=death_text
                    death_city=""
                    death_country=""
            except Exception as e:
                death_year=""
                death_city=""
                death_country=""    
            
            try:            
                gender_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-Spotlight-Spotlight__summaryDetail']//table[@class='app-js-components-Spotlight-Spotlight__info']//tr[td[contains(text(), 'Gender')]]/td[2]")
                gender = gender_element.text.strip()
            except Exception as e:               
                gender=""
                
            try:    
                nationality_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-Spotlight-Spotlight__summaryDetail']//table[@class='app-js-components-Spotlight-Spotlight__info']//tr[td[contains(text(), 'Nationality')]]/td[2]")
                nationality1 = nationality_element.text.strip()       
                nationality = nationality1.replace('\n', ', ') 
            except Exception as e: 
                nationality=""
            
            try:        
                movement_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-Spotlight-Spotlight__summaryDetail']//table[@class='app-js-components-Spotlight-Spotlight__info']//tr[td[contains(text(), 'Movement')]]/td[2]")
                movement1 = movement_element.text.strip()           
                movement = movement1.replace('\n', ', ')   
            except Exception as e: 
                movement=""    
            
            try:
                media_element =driver.find_element(By.XPATH, "//div[@class='app-js-components-Spotlight-Spotlight__summaryDetail']//table[@class='app-js-components-Spotlight-Spotlight__info']//tr[td[contains(text(), 'Media')]]/td[2]")
                media1 = media_element.text.strip()
                media = media1.replace('\n', ', ')
            except Exception as e: 
                media="" 
                    
            try:       
                period_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-Spotlight-Spotlight__summaryDetail']//table[@class='app-js-components-Spotlight-Spotlight__info']//tr[td[contains(text(), 'Period')]]/td[2]")
                period1 = period_element.text.strip()
                period = period1.replace('\n', ', ')            
            except Exception as e: 
                period="" 
            
            try:                
                ranking_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-ColorDisplay-ColorDisplay__container app-js-styles-shared-ScrollContainer__ranking app-js-styles-shared-ScrollContainer__firstColumn']//a")
                ranking = ranking_element.text.strip().split("Global")[0].replace("Top ", "").replace(",", "")             
            except Exception as e: 
                ranking="" 
            
            try:        
                verified_exhibitions_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-ColorDisplay-ColorDisplay__container app-js-styles-shared-ScrollContainer__exhibitions app-js-styles-shared-ScrollContainer__secondColumn']//a")
                verified_exhibitions = verified_exhibitions_element.text.strip()
            except Exception as e: 
                verified_exhibitions="" 
            
            try:        
                solo_exhibitions_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-ColorDisplay-ColorDisplay__container app-js-styles-shared-ScrollContainer__exhibitions app-js-styles-shared-ScrollContainer__secondColumn']//div[@class='app-js-components-ColorDisplay-ColorDisplay__child']//h6[contains(text(), 'Solo')]/following-sibling::a")
                solo_exhibitions = solo_exhibitions_element.text.strip()
            except Exception as e: 
                solo_exhibitions="" 
            
            try:    
                group_exhibitions_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-ColorDisplay-ColorDisplay__container app-js-styles-shared-ScrollContainer__exhibitions app-js-styles-shared-ScrollContainer__secondColumn']//div[@class='app-js-components-ColorDisplay-ColorDisplay__child']//h6[contains(text(), 'Group')]/following-sibling::a")
                group_exhibitions = group_exhibitions_element.text.strip()
            except Exception as e: 
                group_exhibitions="" 
            
            try:    
                biennials_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-ColorDisplay-ColorDisplay__container app-js-styles-shared-ScrollContainer__exhibitions app-js-styles-shared-ScrollContainer__secondColumn']//div[@class='app-js-components-ColorDisplay-ColorDisplay__child']//h6[contains(text(), 'Biennials')]/following-sibling::a")
                biennials = biennials_element.text.strip()
            except Exception as e: 
                biennials="" 
            
            try:
                art_fairs_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-ColorDisplay-ColorDisplay__container app-js-styles-shared-ScrollContainer__exhibitions app-js-styles-shared-ScrollContainer__secondColumn']//div[@class='app-js-components-ColorDisplay-ColorDisplay__child']//h6[contains(text(), 'Art Fairs')]/following-sibling::a")
                art_fairs = art_fairs_element.text.strip()
            except Exception as e: 
                art_fairs=""   
            
            try:              
                exhibitions_top_country_1_element = driver.find_element(By.XPATH, "//div[contains(@class, 'app-js-styles-shared-ScrollContainer__thirdColumn') and contains(@class, 'app-js-components-Spotlight-SpotlightList__spotlightList')]//h5[text()='Most Exhibitions In']/following-sibling::ol/li[1]/a")
                exhibitions_top_country_1 = exhibitions_top_country_1_element.text.strip()
            except Exception as e: 
                exhibitions_top_country_1=""      
            
            try:
                exhibitions_top_country_2_element = driver.find_element(By.XPATH, "//div[contains(@class, 'app-js-styles-shared-ScrollContainer__thirdColumn') and contains(@class, 'app-js-components-Spotlight-SpotlightList__spotlightList')]//h5[text()='Most Exhibitions In']/following-sibling::ol/li[2]/a")
                exhibitions_top_country_2 = exhibitions_top_country_2_element.text.strip()
            except Exception as e: 
                exhibitions_top_country_2=""  
                
            try:
                exhibitions_top_country_3_element =driver.find_element(By.XPATH, "//div[contains(@class, 'app-js-styles-shared-ScrollContainer__thirdColumn') and contains(@class, 'app-js-components-Spotlight-SpotlightList__spotlightList')]//h5[text()='Most Exhibitions In']/following-sibling::ol/li[3]/a")
                exhibitions_top_country_3 = exhibitions_top_country_3_element.text.strip()
            except Exception as e: 
                exhibitions_top_country_3=""      
                
            try:              
                exhibitions_top_country_1_no_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__container']//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__lists']//div[contains(@class, 'app-js-components-MostExhibitedList-MostExhibitedList__container') and contains(@class, 'app-js-components-MostExhibitedWidget-MostExhibitedWidget__list')][1]//div//ol//li[1]")
                exhibitions_top_country_1_no = exhibitions_top_country_1_no_element.text.strip().split('\n')[1]
            except Exception as e: 
                exhibitions_top_country_1_no=""      
            try:
                exhibitions_top_country_2_no_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__container']//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__lists']//div[contains(@class, 'app-js-components-MostExhibitedList-MostExhibitedList__container') and contains(@class, 'app-js-components-MostExhibitedWidget-MostExhibitedWidget__list')][1]//div//ol//li[2]")
                exhibitions_top_country_2_no = exhibitions_top_country_2_no_element.text.strip().split('\n')[1]
            except Exception as e: 
                exhibitions_top_country_2_no=""  
                
            try:
                exhibitions_top_country_3_no_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__container']//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__lists']//div[contains(@class, 'app-js-components-MostExhibitedList-MostExhibitedList__container') and contains(@class, 'app-js-components-MostExhibitedWidget-MostExhibitedWidget__list')][1]//div//ol//li[3]")
                exhibitions_top_country_3_no = exhibitions_top_country_3_no_element.text.strip().split('\n')[1]
            except Exception as e: 
                exhibitions_top_country_3_no=""
                
            try:              
                exhibitions_most_at_1_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__container']//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__lists']//div[contains(@class, 'app-js-components-MostExhibitedList-MostExhibitedList__container') and contains(@class, 'app-js-components-MostExhibitedWidget-MostExhibitedWidget__list')][2]//div//ol//li[1]")
                exhibitions_most_at_1 = exhibitions_most_at_1_element.text.strip().split('\n')[0]
                exhibitions_most_at_1_no = exhibitions_most_at_1_element.text.strip().split('\n')[1]
            except Exception as e: 
                exhibitions_most_at_1=""  
                exhibitions_most_at_1_no=""

            try:              
                exhibitions_most_at_2_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__container']//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__lists']//div[contains(@class, 'app-js-components-MostExhibitedList-MostExhibitedList__container') and contains(@class, 'app-js-components-MostExhibitedWidget-MostExhibitedWidget__list')][2]//div//ol//li[2]")
                exhibitions_most_at_2 = exhibitions_most_at_2_element.text.strip().split('\n')[0]
                exhibitions_most_at_2_no = exhibitions_most_at_2_element.text.strip().split('\n')[1]
            except Exception as e: 
                exhibitions_most_at_2=""  
                exhibitions_most_at_2_no=""
            
            try:              
                exhibitions_most_at_3_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__container']//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__lists']//div[contains(@class, 'app-js-components-MostExhibitedList-MostExhibitedList__container') and contains(@class, 'app-js-components-MostExhibitedWidget-MostExhibitedWidget__list')][2]//div//ol//li[3]")
                exhibitions_most_at_3 = exhibitions_most_at_3_element.text.strip().split('\n')[0]
                exhibitions_most_at_3_no = exhibitions_most_at_3_element.text.strip().split('\n')[1]
            except Exception as e: 
                exhibitions_most_at_3=""  
                exhibitions_most_at_3_no=""
            
            try:              
                exhibitions_most_with_1_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__container']//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__lists']//div[contains(@class, 'app-js-components-MostExhibitedList-MostExhibitedList__container') and contains(@class, 'app-js-components-MostExhibitedWidget-MostExhibitedWidget__list')][3]//div//ol//li[1]")
                exhibitions_most_with_1 = exhibitions_most_with_1_element.text.strip().split('\n')[0]
                exhibitions_most_with_rank_1 = exhibitions_most_with_1_element.text.strip().split('\n')[1].replace("Top ","").replace(",","")
                exhibitions_most_with_1_no = exhibitions_most_with_1_element.text.strip().split('\n')[2]
            except Exception as e: 
                exhibitions_most_with_1=""
                exhibitions_most_with_rank_1=""  
                exhibitions_most_with_1_no=""
            
            try:              
                exhibitions_most_with_2_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__container']//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__lists']//div[contains(@class, 'app-js-components-MostExhibitedList-MostExhibitedList__container') and contains(@class, 'app-js-components-MostExhibitedWidget-MostExhibitedWidget__list')][3]//div//ol//li[2]")
                exhibitions_most_with_2 = exhibitions_most_with_2_element.text.strip().split('\n')[0]
                exhibitions_most_with_rank_2 = exhibitions_most_with_2_element.text.strip().split('\n')[1].replace("Top ","").replace(",","")
                exhibitions_most_with_2_no = exhibitions_most_with_2_element.text.strip().split('\n')[2]
            except Exception as e: 
                exhibitions_most_with_2=""
                exhibitions_most_with_rank_2=""  
                exhibitions_most_with_2_no=""

            try:              
                exhibitions_most_with_3_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__container']//div[@class='app-js-components-MostExhibitedWidget-MostExhibitedWidget__lists']//div[contains(@class, 'app-js-components-MostExhibitedList-MostExhibitedList__container') and contains(@class, 'app-js-components-MostExhibitedWidget-MostExhibitedWidget__list')][3]//div//ol//li[3]")
                exhibitions_most_with_3 = exhibitions_most_with_3_element.text.strip().split('\n')[0]
                exhibitions_most_with_rank_3 = exhibitions_most_with_3_element.text.strip().split('\n')[1].replace("Top ","").replace(",","")
                exhibitions_most_with_3_no = exhibitions_most_with_3_element.text.strip().split('\n')[2]
            except Exception as e: 
                exhibitions_most_with_3=""
                exhibitions_most_with_rank_3=""  
                exhibitions_most_with_3_no=""
            
            try:
                notable_exhibitions_at_1_element = driver.find_element(By.XPATH, "//div[@class='app-js-styles-shared-ScrollContainer__fourthColumn app-js-components-Spotlight-SpotlightList__spotlightList']//h5[text()='Notable Exhibitions At']/following-sibling::ol/li[1]/a")
                notable_exhibitions_at_1 = notable_exhibitions_at_1_element.text.strip()
            except Exception as e: 
                notable_exhibitions_at_1=""      
            
            try:
                notable_exhibitions_at_2_element = driver.find_element(By.XPATH, "//div[@class='app-js-styles-shared-ScrollContainer__fourthColumn app-js-components-Spotlight-SpotlightList__spotlightList']//h5[text()='Notable Exhibitions At']/following-sibling::ol/li[2]/a")
                notable_exhibitions_at_2 = notable_exhibitions_at_2_element.text.strip()
            except Exception as e: 
                notable_exhibitions_at_2=""  
            
            try:
                notable_exhibitions_at_3_element = driver.find_element(By.XPATH, "//div[@class='app-js-styles-shared-ScrollContainer__fourthColumn app-js-components-Spotlight-SpotlightList__spotlightList']//h5[text()='Notable Exhibitions At']/following-sibling::ol/li[3]/a")
                notable_exhibitions_at_3 = notable_exhibitions_at_3_element.text.strip()
            except Exception as e: 
                notable_exhibitions_at_3=""  
            try:
                p_element = driver.find_element(By.XPATH, "//div[@class='app-js-components-AboutWidget-AboutWidget__statement']//div[@class='app-js-components-AboutWidget-AboutWidget__aboutText']//p//span")
                html_content = p_element.get_attribute("outerHTML")
                soup = BeautifulSoup(html_content, "html.parser")
                full_text = ""

                for element in soup.descendants:
                    if isinstance(element, str):
                        full_text += element

                description = full_text.strip().replace("\n", " ").split("For a complete illustration of the")[0] 
            except Exception as e: 
                description=""   
            
                
            artist_data = {
                "Name": artist_name,
                "Birth Year": birth_year,
                "Birth City": birth_city,
                "Birth Country": birth_country,
                "Death Year": death_year,
                "Death City": death_city,
                "Death Country": death_country,
                "Gender": gender,
                "Nationality": nationality,
                "Movement": movement,
                "Media": media,
                "Period": period,
                "Ranking": ranking,
                "Verified Exhibitions": verified_exhibitions,
                "Solo Exhibitions": solo_exhibitions,
                "Group Exhibitions": group_exhibitions,
                "Biennials": biennials,
                "Art Fairs": art_fairs,
                "Exhibitions Top Country 1": exhibitions_top_country_1,
                "Exhibitions Top Country 2": exhibitions_top_country_2,
                "Exhibitions Top Country 3": exhibitions_top_country_3,
                "Exhibitions Top Country 1 No": exhibitions_top_country_1_no,
                "Exhibitions Top Country 2 No": exhibitions_top_country_2_no,
                "Exhibitions Top Country 3 No": exhibitions_top_country_3_no,
                "Exhibitions Most At 1": exhibitions_most_at_1,
                "Exhibitions Most At 2": exhibitions_most_at_2,
                "Exhibitions Most At 3": exhibitions_most_at_3,
                "Exhibitions Most At 1 No": exhibitions_most_at_1_no,
                "Exhibitions Most At 2 No": exhibitions_most_at_2_no,
                "Exhibitions Most At 3 No": exhibitions_most_at_3_no,
                "Exhibitions Most With 1": exhibitions_most_with_1,
                "Exhibitions Most With 2": exhibitions_most_with_2,
                "Exhibitions Most With 3": exhibitions_most_with_3,
                "Exhibitions Most With Rank 1": exhibitions_most_with_rank_1,
                "Exhibitions Most With Rank 2": exhibitions_most_with_rank_2,
                "Exhibitions Most With Rank 3": exhibitions_most_with_rank_3,
                "Exhibitions Most With 1 No": exhibitions_most_with_1_no,
                "Exhibitions Most With 2 No": exhibitions_most_with_2_no,
                "Exhibitions Most With 3 No": exhibitions_most_with_3_no,
                "Notable Exhibitions At 1": notable_exhibitions_at_1,
                "Notable Exhibitions At 2": notable_exhibitions_at_2,
                "Notable Exhibitions At 3": notable_exhibitions_at_3,
                "Description": description
            }
            
            artist_df = pd.DataFrame([artist_data])  
            print(artist_df.to_string())
            try:
                file_exists = os.path.isfile(CSV_FILENAME_OUTPUT)
                artist_df.to_csv(CSV_FILENAME_OUTPUT, sep=';', encoding='utf-8', index=False, mode='a', header=not file_exists)
                print("File saved successfully.")
            except Exception as e:
                print("Error:", e)                
        except Exception as e:
            print("Error:", e)
            
        artist = ArtFactsArtist(artist_name, birth_year,birth_city,birth_country, death_year,death_city,death_country, gender, nationality, movement, media, period, ranking, verified_exhibitions, solo_exhibitions, group_exhibitions, biennials, art_fairs,
                    exhibitions_top_country_1,exhibitions_top_country_2,exhibitions_top_country_3,exhibitions_top_country_1_no,exhibitions_top_country_2_no,exhibitions_top_country_3_no,
                    exhibitions_most_at_1, exhibitions_most_at_2, exhibitions_most_at_3, exhibitions_most_at_1_no, exhibitions_most_at_2_no, exhibitions_most_at_3_no,
                    exhibitions_most_with_1, exhibitions_most_with_2, exhibitions_most_with_3, exhibitions_most_with_rank_1, exhibitions_most_with_rank_2, exhibitions_most_with_rank_3, 
                    exhibitions_most_with_1_no, exhibitions_most_with_2_no, exhibitions_most_with_3_no, notable_exhibitions_at_1,notable_exhibitions_at_2,notable_exhibitions_at_3, description)
        
        # print(artist)
        return driver

    def save_artists_information(self, artist_urls, CSV_FILENAME_OUTPUT):
        os.makedirs(os.path.dirname(CSV_FILENAME_OUTPUT), exist_ok=True)
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(options=options)
        
        for url in artist_urls:   
            try:
                self.get_artist_information(driver, url, CSV_FILENAME_OUTPUT)
            except Exception as e:
                print(f"Error processing {url}: {e}")

        driver.quit()
         
        
    def main_artfacts(self, CSV_FILENAME, CSV_OUTPUT, gather_information):               
        if gather_information:
            print("Information from ArtFacts will be collected...")
            df = pd.read_csv(CSV_FILENAME, delimiter=",", encoding='utf-8')
            urls = df["Url"]
            self.save_artists_information(urls, CSV_OUTPUT) 
        else:
            print("Information from ArtFacts will NOT be collected")

