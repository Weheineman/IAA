#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

dimensionList = ["2", "4", "8", "16", "32"]
fileStems = ["diagonal", "paralelo"]
errorType = sys.argv[1]


# Read from file.
inputFile = open(f"NB{errorType}Error.txt", "r")
trainErrorListList = []
testErrorListList = []
for fileStem in fileStems:
    trainErrorList = []
    testErrorList = []
    for dummy in range(len(dimensionList)):
        [d, trainErr, testErr] = [
            float(value) for value in inputFile.readline().split()
        ]
        trainErrorList.append(trainErr)
        testErrorList.append(testErr)
    trainErrorListList.append(trainErrorList)
    testErrorListList.append(testErrorList)


colors = ["tab:blue", "tab:red"]
x = [int(value) for value in dimensionList]

# Plot graph.
for index in range(len(fileStems)):
    plt.semilogx(
        x,
        trainErrorListList[index],
        marker="o",
        linestyle="dashed",
        color=colors[index],
        label=fileStems[index] + " training error",
    )
    plt.semilogx(
        x,
        testErrorListList[index],
        marker="o",
        color=colors[index],
        label=fileStems[index] + " test error",
    )
plt.title(f"NB using {errorType} for dataset: " + str(fileStems))
plt.xticks(x, x)
plt.xlabel("Number of dimensions of the points")
plt.ylabel("Percentage error")
plt.legend()
plt.savefig(fname=f"NB{errorType}Graph")
