import matplotlib.pyplot as plt
import pandas as pd

method_stem = "k_nn"
file_stem = "ssp"
err_df = pd.read_csv(f"{method_stem}_{file_stem}.err")

k_list = err_df["k"].unique()
train_median = []
valid_median = []
test_median = []
for k in k_list:
    median_df = err_df.loc[err_df["k"] == k].median(axis=0)
    train_median.append(median_df["train_err"])
    valid_median.append(median_df["valid_err"])
    test_median.append(median_df["test_err"])

# Plot graph.
plt.plot(
    k_list,
    train_median,
    marker="o",
    color="tab:blue",
    label="training error",
)
plt.plot(
    k_list,
    valid_median,
    marker="o",
    color="tab:olive",
    label="validation error",
)
plt.plot(
    k_list,
    test_median,
    marker="o",
    color="tab:red",
    label="test error",
)

plt.title(f"{method_stem} using {file_stem}")
plt.xticks(range(0, 101, 20))
plt.xlabel("k")
plt.ylabel("mean squared error")
plt.legend()
plt.savefig(fname=f"{method_stem}_{file_stem}_err")
