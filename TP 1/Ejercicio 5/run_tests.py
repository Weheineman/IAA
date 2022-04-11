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
    treeSize: [float]
    trainingError: [float]
    testError: [float]


C = "0.78"
d = "2"
iterationCount = 30
trainingSizes = ["125", "250", "500", "1000", "2000", "4000"]
testSize = "10000"
fileStems = ["diagonal", "paralelo"]
endResultList = []

# Generate data.
for fileStem in fileStems:
    subprocess.run(["./" + fileStem, d, testSize, C])
    subprocess.run(["mv", fileStem + ".data", fileStem + ".test"])
    endResult = EndResult(treeSize=[], trainingError=[], testError=[])
    for setSize in trainingSizes:
        accumResults = [0] * 14
        for iteration in range(iterationCount):
            subprocess.run(["./" + fileStem, d, setSize, C])
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
        endResult.treeSize.append((avgResults[3] + avgResults[10]) / 2)
        endResult.trainingError.append(avgResults[5])
        endResult.testError.append(avgResults[12])
    endResultList.append(endResult)

# Plot generated data.
colors = ["blue", "red", "green"]
x = trainingSizes
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
plt.title("Datasets: diagonal, paralelo")
plt.xticks(x, x)
plt.xlabel("Training set size")
plt.ylabel("Percentage error")
plt.legend()
plt.savefig(fname="errorGraph")

plt.clf()
for index in range(len(fileStems)):
    endResult = endResultList[index]
    plt.semilogx(
        trainingSizes,
        endResult.treeSize,
        marker="o",
        color=colors[index],
        label=fileStems[index],
    )
plt.title("Datasets: diagonal, paralelo")
plt.xticks(trainingSizes, trainingSizes)
plt.xlabel("Training set size")
plt.ylabel("Decision tree node amount")
plt.legend()
plt.savefig(fname="sizeGraph")

# Cleanup.
for fileStem in fileStems:
    subprocess.run(["rm"] + glob.glob(fileStem + ".*"))
