#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

fileStem = sys.argv[1]

# Read from file.
inputFile = open(f"B{fileStem}MedianError.txt", "r")
trainErrorList = []
validErrorList = []
testErrorList = []
binCountList = []
for _dummy in range(200):
    [binCount, trainErr, validErr, testErr] = [
        float(value) for value in inputFile.readline().split()
    ]
    trainErrorList.append(trainErr)
    validErrorList.append(validErr)
    testErrorList.append(testErr)
    binCountList.append(binCount)

x = [int(value) for value in binCountList]
step = 20
start = step - 1

# Plot graph.
plt.plot(
    x[start::step],
    trainErrorList[start::step],
    marker="o",
    color="tab:blue",
    label=fileStem + " training error",
)
plt.plot(
    x[start::step],
    validErrorList[start::step],
    marker="o",
    color="tab:olive",
    label=fileStem + " validation error",
)
plt.plot(
    x[start::step],
    testErrorList[start::step],
    marker="o",
    color="tab:red",
    label=fileStem + " test error",
)
plt.title(f"B using Median for dataset: {fileStem}")
plt.xticks(x[start::step], x[start::step])
plt.xlabel("Amount of bins")
plt.ylabel("Percentage error")
plt.legend()
plt.savefig(fname=f"B{fileStem}MedianGraph")
