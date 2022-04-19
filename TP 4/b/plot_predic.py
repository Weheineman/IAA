import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

method_stem = "tree"
file_stem = "c_2"
# Read data.
data_frame = pd.read_csv(f"{method_stem}_{file_stem}.predic")

color_list = list(mcolors.TABLEAU_COLORS.keys())
for idx, klass in enumerate(data_frame["predic"].unique()):
    idx_list = data_frame["predic"] == klass
    plt.scatter(
        data_frame.loc[idx_list, "x0"],
        data_frame.loc[idx_list, "x1"],
        marker="o",
        color=color_list[idx],
    )
plt.title(f"{method_stem} prediction using {file_stem}")
plt.savefig(f"{method_stem}_{file_stem}")
