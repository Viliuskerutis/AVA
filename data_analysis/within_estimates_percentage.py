import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.data_filter_pipeline import process_after_scraping
from helpers.file_manager import FileManager
from config import CSVS_PATH

data_df = FileManager.read_all_csvs(CSVS_PATH)
data_for_prediction_df = process_after_scraping(data_df)

# Ignore rows where "Estimated Minimum Price" or "Estimated Maximum Price" is 0
filtered_data_df = data_for_prediction_df[
    (data_for_prediction_df["Estimated Minimum Price"] > 0)
    & (data_for_prediction_df["Estimated Maximum Price"] > 0)
]

# Add a column to indicate if "Sold Price" is within the estimated price range
filtered_data_df["Within Range"] = (
    filtered_data_df["Sold Price"] >= filtered_data_df["Estimated Minimum Price"]
) & (filtered_data_df["Sold Price"] <= filtered_data_df["Estimated Maximum Price"])

# Calculate the percentage of paintings where "Sold Price" is within the estimated price range
within_range_count = filtered_data_df["Within Range"].sum()
total_count = filtered_data_df.shape[0]
percentage_within_range = (
    (within_range_count / total_count) * 100 if total_count > 0 else 0
)

print(
    f"Percentage of paintings where Sold Price is within the estimated range: {percentage_within_range:.2f}%"
)

# Save all rows to a CSV file with the "Within Range" column
filtered_data_df[
    ["Sold Price", "Estimated Minimum Price", "Estimated Maximum Price", "Within Range"]
].to_csv("all_data_within_range.csv", index=False, sep=";")
