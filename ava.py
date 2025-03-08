from flask import Flask, jsonify, request
from helpers.file_manager import FileManager
from config import (
    IMAGES_PATH,
    CSVS_PATH
)
from data_processing.data_filter_pipeline import (
    process_for_image_similarity,
    process_for_predictions
)
from main import (
    try_find_most_similar,
    predict_price
)
import os

# Create a Flask application
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'data/images/runtime'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a route for the root endpoint
@app.route('/', methods=['GET'])
def home():
    return "Hello, this is your server responding!"

# Define a route to receive an image file
@app.route('/evaluate', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file locally
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # from main.py
    # First part - if identical image exists, return data (price) of painting
    image_path_dict = FileManager.get_full_image_paths(IMAGES_PATH)
    data_df = FileManager.read_all_csvs(CSVS_PATH)

    # Keep values that have price, remove non-existant images, remove duplicate photo ids, remove low quality
    data_for_similarity_df = process_for_image_similarity(data_df, image_path_dict)

    input_image_path = file_path
    most_similar = try_find_most_similar(
        image_path_dict, data_for_similarity_df, input_image_path
    )

    if most_similar is None:
        # In current implementation it is expected that image from `input_image_path` exists in data folder
        # User input data should be passed trough processing pipeline before use
        data_for_prediction_df = process_for_predictions(data_df)

        # This should be the user painting information from UI (temporarily picking random value)
        random_painting_data = data_for_prediction_df.sample(n=1).iloc[0].to_dict()

        predicted_price = predict_price(data_for_prediction_df, random_painting_data)

    if os.path.exists(file_path):
        os.remove(file_path)
    
    return jsonify({"data": None if most_similar is None else most_similar.to_json(orient='records'), "price": most_similar["Sold Price"].values[0] if most_similar is not None else predicted_price, "message": "Exact image found inside the storage" if most_similar is not None else "Image price evaluated successfully"})

# Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
