#!/usr/bin/python

from operator import add
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import glob
import re
from dataclasses import dataclass


@dataclass
class EndResult:
    trainingError: [float]
    testError: [float]


C = "0.78"
dimensionList = ["2", "4", "8", "16", "32"]
iterationCount = 30
trainingSize = "250"
testSize = "10000"
fileStems = ["diagonal", "paralelo"]
endResultList = []

# Generate data.
for fileStem in fileStems:
    endResult = EndResult(trainingError=[], testError=[])
    for d in dimensionList:
        subprocess.run(["./" + fileStem, d, testSize, C])
        subprocess.run(["mv", fileStem + ".data", fileStem + ".test"])
        accumResults = [0] * 14
        for iteration in range(iterationCount):
            subprocess.run(["./" + fileStem, d, trainingSize, C])
            completed = subprocess.run(
                ["c4.5", "-f", fileStem, "-u"], capture_output=True, text=True
            )
            resultLines = [
                line for line in completed.stdout.split("\n") if "<<" in line
            ]
            resultList = []
            for line in resultLines:
                resultList += [float(number) for number in re.findall("[0-9.]+", line)]
            accumResults = list(map(add, accumResults, resultList))
        avgResults = [result / iterationCount for result in accumResults]
        endResult.trainingError.append(avgResults[5])
        endResult.testError.append(avgResults[12])
    endResultList.append(endResult)

# Plot generated data.
colors = ["blue", "red", "green"]
x = dimensionList
x = [int(value) for value in x]

for index in range(len(fileStems)):
    endResult = endResultList[index]
    plt.semilogx(
        x,
        endResult.trainingError,
        marker="o",
        linestyle="dashed",
        color=colors[index],
        label=fileStems[index] + " training error",
    )
    plt.semilogx(
        x,
        endResult.testError,
        marker="o",
        color=colors[index],
        label=fileStems[index] + " test error",
    )
plt.title("Datasets: " + str(fileStems))
plt.xticks(x, x)
plt.xlabel("Number of dimensions of the points")
plt.ylabel("Percentage error")
plt.legend()
plt.savefig(fname="errorGraph")

# Cleanup.
for fileStem in fileStems:
    subprocess.run(["rm"] + glob.glob(fileStem + ".*"))
