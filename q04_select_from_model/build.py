# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    np.random.seed(9)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    rf = RandomForestClassifier()
    rf.fit(X,y)
    feature_name = []
    selected_features = SelectFromModel(rf,prefit=True).get_support()
    for col in list(zip(X.columns, selected_features)):
        if(col[1]==True):
            feature_name.append(col[0])
    return feature_name




