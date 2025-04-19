import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import numpy as np
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_COMBINED_DATA_PATH, PROCESSED_DATA_PATH
from helpers.file_manager import FileManager


df = FileManager.read_single_csv(PROCESSED_COMBINED_DATA_PATH, separator=";")

numeric_columns = [
    "Artist Birth Year",
    "Artist Death Year",
    "Creation Year",
    "Width",
    "Height",
    "Sold Price",
    "Estimated Minimum Price",
    "Estimated Maximum Price",
    "Years from auction till now",
    "Years from creation till auction",
    "Years from birth till creation",
    "Artist Lifetime",
    "Artist Years Now",
    "Area",
    "Average Sold Price",
    "Min Sold Price",
    "Max Sold Price",
    "Ranking",
    "Verified Exhibitions",
    "Solo Exhibitions",
    "Group Exhibitions",
    "Biennials",
    "Art Fairs",
    "Exhibitions Top Country 1 No",
    "Exhibitions Top Country 2 No",
    "Exhibitions Top Country 3 No",
    "Exhibitions Most At 1 No",
    "Exhibitions Most At 2 No",
    "Exhibitions Most At 3 No",
    "Exhibitions Most With Rank 1",
    "Exhibitions Most With Rank 2",
    "Exhibitions Most With Rank 3",
    "Exhibitions Most With 1 No",
    "Exhibitions Most With 2 No",
    "Exhibitions Most With 3 No",
    "count",
    "Total Auction Sold Price",
    "Average Auction Sold Price",
    "Auction Record Price",
    "Auction Record Year",
    "Global Market Rank",
    "Influence Score",
    "Popularity Score",
    "Exhibition Prestige Rank",
    "Historical Importance",
    "Average Annual Sales Volume",
    "Top Sale Price Record",
    "Average Hammer Price Deviation Percentage",
    "Average Time to Sell",
    "Repeat Buyer Percentage",
    "Average Seller Premium Percentage",
    "Auction House Global Rank",
    "Prestige Rank",
    "Specialization Score",
    "International Presence Rank",
    "Marketing Power Rank",
    "Average Buyer Competition Score",
    "Liquidity Score",
]
# surface_columns = [col for col in df.columns if col.startswith("Surface_")]
# material_columns = [col for col in df.columns if col.startswith("Materials_")]
image_feature_columns = [col for col in df.columns if col.startswith("Image_Feature_")]
numeric_columns.extend(image_feature_columns)
# numeric_columns.extend(surface_columns)
# numeric_columns.extend(material_columns)

textual_columns = [
    "Artist name",
    "Auction House",
    "Auction Country",
    "Signed",
    "Birth City",
    "Birth Country",
    "Death City",
    "Death Country",
    "Gender",
    "Nationality",
    "Movement",
    "Media",
    "Period",
    "Verified Exhibitions",
    "Group Exhibitions",
    "Exhibitions Top Country 1",
    "Exhibitions Top Country 2",
    "Exhibitions Top Country 3",
    "Exhibitions Most At 1",
    "Exhibitions Most At 2",
    "Exhibitions Most At 3",
    "Exhibitions Most With 1",
    "Exhibitions Most With 2",
    "Exhibitions Most With 3",
    "Notable Exhibitions At 1",
    "Notable Exhibitions At 2",
    "Notable Exhibitions At 3",
    "Description",
    "Artistic Styles",
    "Main Artistic Style",
    "Additional Occupations",
    "Medium Preferences",
    "Primary Market",
    "Auction Type Preference",
]

existing_numeric_columns = df.columns.intersection(numeric_columns).tolist()
numeric_data = df[existing_numeric_columns]
numeric_array = numeric_data.to_numpy()
print("Numeric Data Shape:", numeric_array.shape)

print("Correlation with Sold Price (sorted)")
corr_matrix = df[existing_numeric_columns].corr()
sold_price_corr = corr_matrix["Sold Price"].apply(abs).sort_values(ascending=False)
print(sold_price_corr)


X = df[existing_numeric_columns].drop(columns=["Sold Price"])
y = df["Sold Price"]
# X = X.fillna(X.median())
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X, y)

# print("\nFeature importance (sorted)")
# importances = model.feature_importances_
# feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
# feature_importance_df = feature_importance_df.sort_values(
#     by="Importance", ascending=False
# )
# print(feature_importance_df)

text_model = SentenceTransformer("all-MPNet-base-v2")


def calculate_textual_column_correlation(
    df, textual_columns, target_column="Sold Price", pca_components=None
):
    correlations = []
    for column in textual_columns:
        if column in df.columns:
            print(f"Embedding column: {column}")
            embeddings = text_model.encode(
                df[column].astype(str).tolist(), show_progress_bar=True
            )

            if pca_components:
                # Apply PCA to reduce dimensionality
                print(
                    f"Applying PCA with {pca_components} components for column: {column}"
                )
                pca = PCA(n_components=pca_components)
                reduced_embeddings = pca.fit_transform(embeddings)

                # Calculate correlation for each PCA component
                target_values = df[target_column].values
                for i in range(reduced_embeddings.shape[1]):
                    component_correlation, _ = pearsonr(
                        reduced_embeddings[:, i], target_values
                    )
                    correlations.append(
                        (f"{column}_PCA_Component_{i+1}", component_correlation)
                    )
            else:
                # # Use mean of embeddings if PCA is not specified
                # reduced_embeddings = np.mean(embeddings, axis=1)
                # target_values = df[target_column].values
                # correlation, _ = pearsonr(reduced_embeddings, target_values)
                # correlations.append((column, correlation))
                # Calculate Pearson correlation for each dimension of the embedding
                target_values = df[target_column].values
                dimension_correlations = []
                for i in range(embeddings.shape[1]):
                    dimension_correlation, _ = pearsonr(embeddings[:, i], target_values)
                    dimension_correlations.append(dimension_correlation)

                # Compute the mean of the correlations across all dimensions
                mean_correlation = np.mean(dimension_correlations)
                correlations.append((column, mean_correlation))

    # Sort correlations by absolute value in descending order
    correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

    print("\nTextual Column Correlations with Sold Price:")
    for column, corr in correlations:
        print(f"{column}: {corr:.4f}")

    return correlations


textual_column_correlations = calculate_textual_column_correlation(
    df, textual_columns, pca_components=None
)
