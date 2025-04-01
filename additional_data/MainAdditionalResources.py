from Artsy import Artsy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ARTSY_CSV_PATH
)

if __name__ == "__main__":
    Artsy().main_artsy(ARTSY_CSV_PATH, gather_information=True)