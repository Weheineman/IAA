import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

file_stem = "BBBs"
estimator_a = SVC(kernel="poly")
estimator_b = SVC(kernel="poly", degree=1)
n_splits = 10
table_value = 2.26 # t_{N, v} for n=95% and v=9
param_grid_a = {"degree": range(1, 5), "C": np.arange(0.5, 10, 0.1)}
param_grid_b = {"C": np.arange(3, 40, 0.5)}

# Read data.
data_frame = pd.read_csv(f"{file_stem}.data", header=None)

features = data_frame.iloc[:, :-1]
target = data_frame.iloc[:, -1]

deltas = []
k_fold = StratifiedKFold(n_splits, shuffle=True)
for train_index, val_index in k_fold.split(features, target):
    errors = []
    for estimator, param_grid in zip([estimator_a, estimator_b], [param_grid_a, param_grid_b]):
        model = GridSearchCV(estimator, param_grid)
        model.fit(features.iloc[train_index], target.iloc[train_index])
        errors.append(1 - model.score(features.iloc[val_index], target.iloc[val_index]))
    deltas.append(errors[0] - errors[1])
    print(f"delta: {deltas[-1]}")

delta_mean = np.mean(deltas)
delta_std = sum([(delta - delta_mean)**2 for delta in deltas])
delta_std /= n_splits*(n_splits-1)
delta_std = np.sqrt(delta_std)

print(f"Delta (mean): {np.mean(deltas)}")
print(f"Delta error: {delta_std * table_value}")
