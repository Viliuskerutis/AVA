import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FILTERED_COMBINED_DATA_PATH

df = pd.read_csv(FILTERED_COMBINED_DATA_PATH)

numerical_features = ['Sold Price', 'Estimated Minimum Price', 'Estimated Maximum Price', 'Area', 'Artist Lifetime',
                      'Width', 'Height', 'Artist Years Now', 'Years from auction till now', 'Years from creation till auction',
                      'Years from birth till creation', 'Min Sold Price', 'Max Sold Price', 'Average Sold Price']

# numerical_features = [
#     'Artist Birth Year', 'Artist Death Year', 'Creation Year', 'Width', 'Height',
#     'Sold Price', 'Estimated Minimum Price', 'Estimated Maximum Price', 'Years from auction till now',
#     'Years from creation till auction', 'Years from birth till creation', 'Artist Lifetime',
#     'Artist Years Now', 'Area', 'Average Sold Price', 'Min Sold Price', 'Max Sold Price',
#     'Ranking', 'Verified Exhibitions', 'Solo Exhibitions', 'Group Exhibitions',
#     'Biennials', 'Art Fairs', 'Exhibitions Top Country 1 No', 'Exhibitions Top Country 2 No',
#     'Exhibitions Top Country 3 No', 'Exhibitions Most At 1 No', 'Exhibitions Most At 2 No',
#     'Exhibitions Most At 3 No', 'Exhibitions Most With Rank 1', 'Exhibitions Most With Rank 2',
#     'Exhibitions Most With Rank 3', 'Exhibitions Most With 1 No', 'Exhibitions Most With 2 No',
#     'Exhibitions Most With 3 No', 'count', 'Total Auction Sold Price', 'Average Auction Sold Price',
#     'Auction Record Price', 'Auction Record Year', 'Global Market Rank', 'Influence Score',
#     'Popularity Score', 'Exhibition Prestige Rank', 'Historical Importance',
#     'Average Annual Sales Volume', 'Top Sale Price Record', 'Average Hammer Price Deviation Percentage',
#     'Average Time to Sell', 'Repeat Buyer Percentage', 'Average Seller Premium Percentage',
#     'Auction House Global Rank', 'Prestige Rank', 'Specialization Score', 'International Presence Rank',
#     'Marketing Power Rank', 'Average Buyer Competition Score', 'Liquidity Score'
# ]

# boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_features])
plt.title('Boxplots of Selected Numerical Features')
plt.xticks(rotation=45)
plt.show()


# correlation
corr_matrix = df[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# overall information max, min etc
print(df[numerical_features].describe())


# VIF (multicollinearity detection)
df_cleaned = df[numerical_features].replace([float('inf'), -float('inf')], float('nan')).dropna()
X = df_cleaned
X = add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
