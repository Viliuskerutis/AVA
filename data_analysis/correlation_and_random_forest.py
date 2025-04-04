import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import RandomForestRegressor
from config import FILTERED_COMBINED_DATA_PATH

df = pd.read_csv(FILTERED_COMBINED_DATA_PATH)

numeric_columns = [
    'Artist Birth Year', 'Artist Death Year', 'Creation Year', 'Width', 'Height',
    'Sold Price', 'Estimated Minimum Price', 'Estimated Maximum Price', 'Years from auction till now',
    'Years from creation till auction', 'Years from birth till creation', 'Artist Lifetime',
    'Artist Years Now', 'Area', 'Average Sold Price', 'Min Sold Price', 'Max Sold Price',
    'Ranking', 'Verified Exhibitions', 'Solo Exhibitions', 'Group Exhibitions',
    'Biennials', 'Art Fairs', 'Exhibitions Top Country 1 No', 'Exhibitions Top Country 2 No',
    'Exhibitions Top Country 3 No', 'Exhibitions Most At 1 No', 'Exhibitions Most At 2 No',
    'Exhibitions Most At 3 No', 'Exhibitions Most With Rank 1', 'Exhibitions Most With Rank 2',
    'Exhibitions Most With Rank 3', 'Exhibitions Most With 1 No', 'Exhibitions Most With 2 No',
    'Exhibitions Most With 3 No', 'count', 'Total Auction Sold Price', 'Average Auction Sold Price',
    'Auction Record Price', 'Auction Record Year', 'Global Market Rank', 'Influence Score',
    'Popularity Score', 'Exhibition Prestige Rank', 'Historical Importance',
    'Average Annual Sales Volume', 'Top Sale Price Record', 'Average Hammer Price Deviation Percentage',
    'Average Time to Sell', 'Repeat Buyer Percentage', 'Average Seller Premium Percentage',
    'Auction House Global Rank', 'Prestige Rank', 'Specialization Score', 'International Presence Rank',
    'Marketing Power Rank', 'Average Buyer Competition Score', 'Liquidity Score'
]

numeric_data = df[numeric_columns]
numeric_array = numeric_data.to_numpy()
print("Numeric Data Shape:", numeric_array.shape)

print("Correlation with Sold Price (sorted)")
corr_matrix = df[numeric_columns].corr()
sold_price_corr = corr_matrix['Sold Price'].sort_values(ascending=False)
print(sold_price_corr)


X = df[numeric_columns].drop(columns=['Sold Price'])
y = df['Sold Price']
# X = X.fillna(X.median())
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("\nFeature importance (sorted)")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)
