import os
import pickle
import random
import pandas as pd
from PIL import Image, UnidentifiedImageError
from typing import Any, Dict, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class FileManager:
    """Handles file operations including reading/writing images, CSVs, and pickle files."""

    @staticmethod
    def save_to_pkl(data: Any, file_path: str) -> None:
        """
        Saves data to a pickle (.pkl) file.

        Args:
            data (Any): The data to be saved.
            file_path (str): Path where the file should be saved.

        Raises:
            IOError: If an error occurs during file writing.
        """
        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        except IOError as e:
            raise IOError(f"Error saving data to {file_path}: {str(e)}")

    @staticmethod
    def load_from_pkl(file_path: str) -> Optional[Any]:
        """
        Loads data from a pickle (.pkl) file.

        Args:
            file_path (str): Path to the pickle file.

        Returns:
            Any: Loaded data, or None if file does not exist or cannot be read.

        Raises:
            IOError: If an error occurs during file reading.
        """
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, IOError) as e:
            raise IOError(f"Error loading data from {file_path}: {str(e)}")

    @staticmethod
    def get_full_image_paths(root_path: str) -> Dict[str, str]:
        """
        Retrieves all image file paths in a directory.

        Args:
            root_path (str): Root directory to scan for images.

        Returns:
            Dict[str, str]: Dictionary where keys are filenames (without extension) and values are full paths.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Directory not found: {root_path}")

        image_dict = {}

        for foldername, _, filenames in os.walk(root_path):
            for filename in filenames:
                file_name, file_extension = os.path.splitext(filename)
                if file_extension.lower() in IMAGE_EXTENSIONS:
                    full_path = os.path.abspath(os.path.join(foldername, filename))
                    image_dict[file_name] = full_path

        return image_dict

    @staticmethod
    def read_single_csv(file_path: str) -> pd.DataFrame:
        """
        Reads a single CSV file and returns a DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the CSV data.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If an error occurs while reading the CSV file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            return pd.read_csv(file_path, sep=";")
        except Exception as e:
            raise IOError(f"Error reading CSV file {file_path}: {str(e)}")

    @staticmethod
    def read_all_csvs(root_path: str) -> pd.DataFrame:
        """
        Reads all CSV files in a directory and combines them into a single DataFrame.

        Args:
            root_path (str): Directory containing CSV files.

        Returns:
            pd.DataFrame: A combined DataFrame from all CSV files. Returns an empty DataFrame if no CSV files are found.

        Raises:
            FileNotFoundError: If the directory does not exist.
            IOError: If an error occurs while reading the CSV files.
        """
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Directory not found: {root_path}")

        csv_files = [
            os.path.join(root_path, file)
            for file in os.listdir(root_path)
            if file.lower().endswith(".csv")
        ]

        if not csv_files:
            return pd.DataFrame()

        dataframes = []
        for file in csv_files:
            dataframes.append(FileManager.read_single_csv(file))

        return pd.concat(dataframes, ignore_index=True)

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """
        Loads an image and converts it to RGB format.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Image.Image: Loaded image in RGB format.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnidentifiedImageError: If the file is not a valid image.
            IOError: If an issue occurs while opening the file.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path)
            return image.convert("RGB") if image.mode != "RGB" else image
        except UnidentifiedImageError:
            raise UnidentifiedImageError(f"Invalid image file: {image_path}")
        except IOError as e:
            raise IOError(f"Error loading image {image_path}: {str(e)}")

    def get_random_image_path(root_folder: str) -> str:
        """
        Picks a random image from the given root folder, including all nested subfolders.

        Args:
            root_folder (str): The path to the root folder where the search will begin.

        Returns:
            str: The full path of a randomly selected image file.

        Raises:
            ValueError: If no images are found in the root folder or its subfolders.
        """
        # Supported image extensions (can be expanded if needed)

        # Collect all image files recursively
        image_files = []
        for folder_path, _, files in os.walk(root_folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(folder_path, file))

        # If no images found, raise an error
        if not image_files:
            raise ValueError(f"No images found in '{root_folder}' or its subfolders.")

        return random.choice(image_files)
