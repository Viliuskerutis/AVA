import os
import pandas as pd

logs_folder = os.path.dirname(__file__)
output_file = os.path.join(logs_folder, "combined_logs.csv")


csv_files = [f for f in os.listdir(logs_folder) if f.endswith(".csv")]

combined_df = pd.concat(
    [pd.read_csv(os.path.join(logs_folder, file)) for file in csv_files]
)

combined_df.to_csv(output_file, index=False)

print(f"Combined CSV saved to {output_file}")
