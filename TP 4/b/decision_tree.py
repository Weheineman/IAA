from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Parameters.
file_stem = "c_2"
col_names = ["x0", "x1", "x2", "x3", "target"]

# Read data.
train = pd.read_csv(f"{file_stem}.data", names=col_names)
test = pd.read_csv(f"{file_stem}.test", names=col_names)

# Train on train, predict on test.
tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(train.iloc[:, :-1], train.iloc[:, -1])
test["predic"] = tree.predict(test.iloc[:, :-1])

# Write predictions to file.
test.to_csv(f"tree_{file_stem}.predic")

test_err = test["target"].ne(test["predic"]).sum() / len(test)
print(f"Decision Tree test error: {test_err * 100}%")
