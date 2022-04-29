import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

file_stem = "BBBs"
estimator = GaussianNB()
n_splits = 10
param_grid = {}

# Read data.
data_frame = pd.read_csv(f"{file_stem}.data", header=None)

features = data_frame.iloc[:, :-1]
target = data_frame.iloc[:, -1]

err_list = []
k_fold = StratifiedKFold(n_splits, shuffle=True)
for train_index, val_index in k_fold.split(features, target):
    model = GridSearchCV(estimator, param_grid)
    model.fit(features.iloc[train_index], target.iloc[train_index])
    val_score = model.score(features.iloc[val_index], target.iloc[val_index])
    err_list.append(1 - val_score)
    print(f"Validation error: {1 - val_score} with params: {model.best_params_}")

print(f"Estimated test error (mean): {np.mean(err_list)}")
print(f"Standard deviation: {np.std(err_list)}")
