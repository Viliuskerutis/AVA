# Code repository for AVA (Annifer) project

## Overview & General Workflow

This project implements a **painting price prediction system** combining **image similarity search** and **price regression prediction** techniques. The system is designed to work in two phases:

### Phase 1: Image Similarity Search
- Given a new input painting image (e.g., uploaded by a user), the system compares it to the paintings already present in the dataset.
- This comparison is performed using **pre-extracted image features** and a configurable **similarity model**.
- If a sufficiently similar painting (above a predefined threshold) exists in the dataset, the system **returns the data of the matched painting**, including:
    - Painting name
    - Artist name
    - Sold price
    - Property list can be extended... (see Data section)

### Phase 2: Price Prediction (if no match found)
- If no sufficiently similar painting is found, or if the matched painting was not sold, the system predicts the price using **regression models**.
- In this case, all available structured data is used to train a regression model (should be provided by user).
- Textual features are embedded using a pre-configured **Sentence Transformer**
- The regression model then predicts the **expected sold price** for the input painting.

### Setup/Development
- Default flake8 settings for formatting during development (setup.cfg contains some changes)
- Preferable virtual environment or conda environment should be used
- To install required packages use `pip install -r requirements.txt`
- `experiments` folders contain code used for experimentation only, this does not need to be deployed

## Data Information

### Data Access
- Full access to the data can be obtained by contacting **Gytis**.
- This data should **not be included in any public repository**, including extracted features.
- GitHub might also limit the uploads due to size.

### Data structure

`data` folder contains three sub-folders:
- `additional_data` - holds temporary files and additional file sources which can be used optionally
- `csvs` - ";" separated UTF-8 csv files containg painting data (see section below)
- `images` - contains sub-folders inside for different data sources of images. Current `FileHandler` implementation reads all directories inside this folder to gather images.

### CSV File Properties
The CSV files inside the `csvs` folder should contain the following columns (in the exact order and with these names):

- **`Artist name`**: Name of the artist who created the painting.
- **`Artist Birth Year`**: Years of artist birth (if available).
- **`Artist Death Year`**:  Year of artist death (if available).
- **`Painting name`**: Title of the painting.
- **`Creation Year`**: Year when the painting was created.
- **`Description`**: Additional textual description of the painting.
- **`Width`**: Physical width of the painting.
- **`Height`**: Physical height of the painting.
- **`Sold Price`**: Actual price at which the painting was sold (if available).
- **`Estimated Minimum Price`**: Estimated minimum price of painting
- **`Estimated Maximum Price`**: Estimated maximum price of painting
- **`Auction name`**: Name of the auction event where the painting was sold.
- **`Auction Date`**: Date when the auction took place.
- **`Auction House`**: Name of the auction house hosting the sale.
- **`Auction City Information`**: City where the auction was held.
- **`Details`**: Additional auction-related details.
- **`Photo id`**: Unique identifier linking the painting entry to an image in the `images/*some_source*` folder.

### Properties Used for Price Prediction (this will get updated)
The following properties are used during the price prediction process. Regarding implementation, these fields can be provided by user directly or engineered from columns found in CSV file descriptions.

- **`Width`**: Physical width of the painting.
- **`Height`**: Physical height of the painting.
- **`Years from auction till now`**: (calculated) Number of years between the auction date and the current date.
- **`Years from creation till auction`**: (calculated) Number of years between the painting's creation year and its auction date.
- **`Artist Lifetime`**: (calculated) Number of years the artist lived.
- **`Average Sold Price`**: (calculated) Average historical price of all paintings by the same artist.
- **`Min Sold Price`**: (calculated) Minimum historical price of all paintings by the same artist (calculated from data).
- **`Max Sold Price`**: (calculated) Maximum historical price of all paintings by the same artist (calculated from data).
- **`Artist name`**: Name of the artist who created the painting.
- **`Auction House`**: Name of the auction house hosting the sale.
- **`Auction Country`**: Country where the auction took place.
- **`Materials`**: Description of the materials used to create the painting (e.g., oil on canvas).
- **`Surface`**: Description of the painting's surface (e.g., canvas, paper).
- **`Signed`**: Indicates if the painting is signed by the artist.

If **artist-related data from external sources** is enabled, the following properties are also included:

- **`Ranking`**: Artist’s ranking within the artfacts dataset.
- **`Birth Country`**: Country where the artist was born.
- **`Gender`**: Gender of the artist.
- **`Nationality`**: Nationality of the artist.
- **`Movement`**: Artistic movement the artist belongs to.
- **`Birth Year`**: Year the artist was born (same as in the CSV files).
- **`Death Year`**: Year the artist died (same as in the CSV files).
- **`Verified Exhibitions`**: Number of verified exhibitions the artist participated in.
- **`Solo Exhibitions`**: Number of solo exhibitions the artist had.
- **`Group Exhibitions`**: Number of group exhibitions the artist participated in.
- **`Biennials`**: Number of biennial exhibitions the artist participated in.
- **`Art Fairs`**: Number of art fairs the artist’s work was shown at.

If **image features are enabled**, additional properties like:
- **`Image_Feature_1`**, **`Image_Feature_2`**, ..., **`Image_Feature_N`** are added after applying dimensionality reduction (PCA) on extracted image features.

## Models Used

### Regressors 
The following regression models are supported or can be easily swapped in the pipeline:
- **LightGBM Regressor** (currently used by default)
- **K-Nearest Neighbors Regressor** 
- **Random Forest Regressor**

These are implemented from packages directly.

### Image Similarity Models
Image similarity is performed using one of the following pre-trained deep learning models:
- **ResNet-50**
- **GoogLeNet**  

These models are used to extract image embeddings (numerical representations of images) to compute pairwise similarity. Models are available trough `torchvision` package - if cached version of model is not present it downloads required architecture and weights automatically.

### Embedding Models
Textual attributes are embedded using one of the following Sentence Transformers:
- **all-MiniLM-L6-v2**
- **all-MPNet-base-v2**

By default, these models are automatically downloaded from Hugging Face if they do not exist in the local cache.
