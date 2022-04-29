import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

file_stem = "dos_elipses"
# Read data.
data_frame = pd.read_csv(f"{file_stem}.predic")

color_list = list(mcolors.TABLEAU_COLORS.keys())
for idx, klass in enumerate(data_frame["predic"].unique()):
    idx_list = data_frame["predic"] == klass
    plt.scatter(
        data_frame.loc[idx_list, "x"],
        data_frame.loc[idx_list, "y"],
        marker="o",
        color=color_list[idx],
    )
plt.savefig(f"{file_stem}")
