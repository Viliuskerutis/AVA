from Artsy import Artsy
from ArtFactsUrls import ArtFactsUrls
from ArtFacts import ArtFacts
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ARTSY_CSV_PATH, ADDITIONAL_DATA_PATH
)

if __name__ == "__main__":
    artfacts_url_path = f"{ADDITIONAL_DATA_PATH}/artfacts_urls.csv"
    artfacts_path = f"{ADDITIONAL_DATA_PATH}/results_artfacts_artists_final.csv"
    Artsy().main_artsy(ARTSY_CSV_PATH, gather_information=True)
    ArtFactsUrls().main_artfacts_url(artfacts_url_path,gather_information=True)
    ArtFacts().main_artfacts(artfacts_url_path, artfacts_path, gather_information=True)
