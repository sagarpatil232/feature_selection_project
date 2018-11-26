# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()


# Your solution code here
def forward_selected(data,model):
    old_r2_score = 0
    new_r2_score = 1
    features = list(data.drop('SalePrice',axis=1).columns)
    selected_features = []
    r2_score_features = []
    X_selected = pd.DataFrame()
    result = pd.DataFrame()
    y = data['SalePrice']
    while(True):
        scores = []
        for i in range(len(features)):
            X = data[features[i]]
            X_selected = result
            X_selected = pd.concat([X_selected,X], axis=1)
            model.fit(X_selected,y)
            y_pred = model.predict(X_selected)
            scores.append(r2_score(y,y_pred))
            X_selected = result
            np_scores = np.array(scores)
        new_r2_score = np_scores.max()
        if(new_r2_score>old_r2_score):
            old_r2_score=new_r2_score
            result = pd.concat([result,data[features[np.argmax(np_scores)]]], axis=1)
            data = data.drop(features[np.argmax(np_scores)],axis = 1)
            selected_features.append(features[np.argmax(np_scores)])
            r2_score_features.append(new_r2_score)
            features.remove(features[np.argmax(np_scores)])
        else:
            break
    return selected_features,r2_score_features
# X = data.drop('SalePrice',1)
# y = data.iloc[:,-1]
# features = X.columns
# r2_scores = []
# for feature in list(features):
#     df = X.loc[:,[feature]]
#     model.fit(df,y)
#     y_pred = model.predict(df) 
#     r2_scores.append((feature,r2_score(y, y_pred)))
# max = r2_scores[0][1]
# max_feature = r2_scores[0][0]
#  = []
# while(len(r2_scores_sorted)!=len(r2_scores)):
#     for item in r2_scores:
#         if(max < item[1]):
#             max = item[1]
#             r2_scores_sorted.append(item)
# max_feature
# #data.head()
# #model.set_params()
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]
# flag = True
# #print(X.columns)
# features = X.columns
# r2_scores = []
# print('features')
# for feature in list(features):
#     X = pd.DataFrame(X[feature])
#     model.fit(X,y)
#     y_pred = model.predict(X) 
#     r2_scores.append(r2_score(y, y_pred))
# print(r2_scores)
# # while(flag==True):
# #     for feature in features:
# #         X = X[[feature]]
# #         model.fit(X,y)
# #         y_pred = model.predict(X) 
# #         y_pred = r2_score(y, y_pred)
# # print(y_pred)
# X.columns
# data.head()
# model.set_params()
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]
# flag = True
# features = X.columns
# r2_scores = 
# while(flag==True):
#     for feature in features:
#         X = X[[feature]]
#         model.fit(X,y)
#         y_pred = model.predict(X) 
#         y_pred = r2_score(y, y_pred)
# print(y_pred)
# X[['GrLivArea','GarageArea']]



