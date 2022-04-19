import matplotlib.pyplot as plt
import pandas as pd

dimensionList = ["2", "4", "8", "16", "32"]
fileStems = ["diagonal", "paralelo"]
suffix = "opt"

df = pd.read_csv(f"KNNMedian_{suffix}.err")

colors = ["tab:blue", "tab:red"]
x = [int(value) for value in dimensionList]

# Plot graph.
for index in range(len(fileStems)):
    l = index * len(dimensionList)
    r = (index + 1) * len(dimensionList)
    plt.semilogx(
        x,
        (df.iloc[l:r])["train_err"],
        marker="o",
        linestyle="dashed",
        color=colors[index],
        label=fileStems[index] + " training error",
    )
    plt.semilogx(
        x,
        (df.iloc[l:r])["test_err"],
        marker="o",
        color=colors[index],
        label=fileStems[index] + " test error",
    )
plt.title(f"Median KNN_{suffix} error for datasets:" + str(fileStems))
plt.xticks(x, x)
plt.xlabel("Number of dimensions of the points")
plt.ylabel("Percentage error")
plt.legend()
plt.savefig(fname=f"KNN_{suffix}MedianGraph")
