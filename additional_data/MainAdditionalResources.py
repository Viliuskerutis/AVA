from Artsy import Artsy
from ArtFactsUrls import ArtFactsUrls
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ARTSY_CSV_PATH, ADDITIONAL_DATA_PATH
)

if __name__ == "__main__":
    artfacts_path = f"{ADDITIONAL_DATA_PATH}/artfact_urls.csv"
    Artsy().main_artsy(ARTSY_CSV_PATH, gather_information=True)
    ArtFactsUrls().main_artfacts_url(artfacts_path,gather_information=True)
