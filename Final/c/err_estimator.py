import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

estimator = DecisionTreeClassifier(criterion="entropy")
n_splits = 10
param_grid = {}

digits = load_digits()
features = digits.data
target = digits.target

err_list = []
k_fold = StratifiedKFold(n_splits, shuffle=True)
for train_index, val_index in k_fold.split(features, target):
    model = GridSearchCV(estimator, param_grid)
    model.fit(features[train_index], target[train_index])
    val_score = model.score(features[val_index], target[val_index])
    err_list.append(1 - val_score)
    print(f"Validation error: {1 - val_score} with params: {model.best_params_}")

print(f"Estimated test error (mean): {np.mean(err_list)}")
print(f"Standard deviation: {np.std(err_list)}")
