# Insurance 
# https://www.kaggle.com/datasets/awaiskaggler/insurance-csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('Machine Learning 3/insurance.csv')
print(df)

text_cols = []
numerical_cols = []
categories = []
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        if len(df[col].value_counts()) <= 10:
            categories.append(col)
        numerical_cols.append(col)
    else:
        if len(df[col].value_counts()) <= 10:
            categories.append(col)
        text_cols.append(col)

df.drop_duplicates(inplace=True)
# onehotencode the categories
ct = ColumnTransformer([
    ('onehot', OneHotEncoder(sparse_output=False), categories)
], remainder='passthrough')

transformed_data = ct.fit_transform(df)

new_df = pd.DataFrame(data=transformed_data)

new_df.rename(columns={16 : 'Target'}, inplace=True)

# modelling, calculating cross val scores then averaging. using X=X, y=y since it automatically uses its own Train/Test split
X = new_df.drop(columns=['Target'])
y = new_df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

catboost_model_scores = cross_val_score(estimator=CatBoostRegressor(), cv=7,scoring='r2', X=X, y=y)
xgb_model_scores = cross_val_score(estimator=XGBRegressor(), cv=7, scoring='r2', X=X, y=y)
lasso_model_scores = cross_val_score(estimator=Lasso(), cv=7, scoring='r2', X=X, y=y)
elasticNet_model_scores = cross_val_score(estimator=ElasticNet(), cv=7, scoring='r2', X=X, y=y)
ridge_model_scores = cross_val_score(estimator=Ridge(), cv=7, scoring='r2', X=X, y=y)
svr_model_scores = cross_val_score(estimator=SVR(kernel='linear'), cv=7, scoring='r2', X=X, y=y)
nuSVR_model_scores = cross_val_score(estimator=NuSVR(), cv=7, scoring='r2', X=X, y=y)
forest_model_scores = cross_val_score(estimator=RandomForestRegressor(), cv=7, scoring='r2', X=X, y=y)

# use loop to fit, get scores
models_score = [catboost_model_scores, xgb_model_scores, lasso_model_scores, elasticNet_model_scores, ridge_model_scores, svr_model_scores, nuSVR_model_scores, forest_model_scores]
models_str = ['catboost_model', 'xgb_model', 'lasso_model', 'elasticNet_model', 'ridge_model', 'svr_model', 'nuSVR_model', 'forest_model']
mean_scores = []
for score in models_score:
    mean_scores.append(np.mean(score))

# scores plot
fig, ax = plt.subplots()
bars = ax.bar(models_str, mean_scores, color=['springgreen', 'paleturquoise', 'lightcoral', 'sandybrown', 'darkseagreen', 'royalblue', 'orchid', 'lightpink'])
ax.set_yticks(np.arange(-0.2, 1.1, 0.1))
ax.set(xlabel='Different Models', ylabel='R2 Scores (CV=7)', title='Training Scores CV=7')

handles = []
for bar in bars:
    handles.append(Line2D([0], [0], linewidth=5, color=bar.get_facecolor()))
ax.legend(handles=handles, labels=models_str, loc='lower right')

for p in bars.patches:
    ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()), xytext=(10, 10), va='center', ha='center', textcoords='offset points')

ax.axhline(linestyle='-', color='Black')

plt.show()
