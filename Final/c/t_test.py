import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

estimator_a = GaussianNB()
estimator_b = DecisionTreeClassifier(criterion="entropy")
n_splits = 10
table_value = 2.26 # t_{N, v} for n=95% and v=9
param_grid_a = {}
param_grid_b = {}

# Read data.
digits = load_digits()
features = digits.data
target = digits.target

deltas = []
k_fold = StratifiedKFold(n_splits, shuffle=True)
for train_index, val_index in k_fold.split(features, target):
    errors = []
    for estimator, param_grid in zip([estimator_a, estimator_b], [param_grid_a, param_grid_b]):
        model = GridSearchCV(estimator, param_grid)
        model.fit(features[train_index], target[train_index])
        errors.append(1 - model.score(features[val_index], target[val_index]))
    deltas.append(errors[0] - errors[1])
    print(f"delta: {deltas[-1]}")

delta_mean = np.mean(deltas)
delta_std = sum([(delta - delta_mean)**2 for delta in deltas])
delta_std /= n_splits*(n_splits-1)
delta_std = np.sqrt(delta_std)

print(f"Delta (mean): {np.mean(deltas)}")
print(f"Delta error: {delta_std * table_value}")
