import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FILTERED_COMBINED_DATA_PATH

df = pd.read_csv(FILTERED_COMBINED_DATA_PATH)
numeric_features = df.select_dtypes(include=[np.number]).drop(columns=["Sold Price"], errors='ignore')
target = df["Sold Price"].dropna()

numeric_features = numeric_features.loc[target.index].dropna()
target = target.loc[numeric_features.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_features)

lasso = LassoCV(cv=10, random_state=42).fit(X_scaled, target)
lasso_importance = pd.Series(np.abs(lasso.coef_), index=numeric_features.columns)
lasso_importance_sorted = lasso_importance.sort_values(ascending=False)
print(lasso_importance_sorted[lasso_importance_sorted > 0])
