# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    rf = RandomForestClassifier()
    rf.fit(X,y)
    nos= int(len(X.columns)/2)
    rfe = RFE(rf, n_features_to_select=nos)
    rfe = rfe.fit(X, y)
    top_features = []
    for t in list(zip(rfe.ranking_,X.columns)):
        if t[0]==1:
            top_features.append(t[1])
    top_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF',
                    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea',
                    'WoodDeckSF', 'OpenPorchSF', 'YrSold']    
    return top_features








