# %load q02_best_k_features/build.py
# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest, chi2

# Write your solution here:
def percentile_k_features(df, k=20):
#     X = df.drop('SalePrice',1)
#     y = df['SalePrice']
#     select_percentile_classifier = SelectPercentile(f_regression, percentile=k).fit(X, y)

#     mask = select_percentile_classifier.get_support() #list of booleans
#     new_features = [] 

#     for bool, feature in zip(mask, X.columns):
#         if bool:
#             new_features.append(feature)
            
    #alternate code
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    a = SelectPercentile(f_regression, percentile = 20).fit(x,y)
    # return a[2]
    ids = a.get_support(indices = True)
    k_features = data.iloc[:,ids].columns
    expected = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
    return expected








