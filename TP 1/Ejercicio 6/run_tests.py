#!/usr/bin/python

from operator import add
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import glob
import re
from dataclasses import dataclass
from ideal_classifier import run_ideal_classifier


@dataclass
class EndResult:
    beforePruningTestError: [float]
    afterPruningTestError: [float]


cList = [str(value) for value in np.arange(0.5, 3, 0.5)]
d = "5"
iterationCount = 20
trainingSize = "250"
testSize = "10000"
fileStems = ["diagonal", "paralelo"]
endResultList = []

# Generate data.
idealResult = [[], []]
for index in range(len(fileStems)):
    fileStem = fileStems[index]
    endResult = EndResult(beforePruningTestError=[], afterPruningTestError=[])
    for C in cList:
        subprocess.run(["./" + fileStem, d, testSize, C])
        subprocess.run(["mv", fileStem + ".data", fileStem + ".test"])
        idealResult[index].append(run_ideal_classifier(fileStem + ".test"))
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
        endResult.beforePruningTestError.append(avgResults[9])
        endResult.afterPruningTestError.append(avgResults[12])
    endResultList.append(endResult)

print(idealResult)

# Plot generated data.
colors = ["blue", "red", "green"]
# fileStems.append("ideal")
x = cList
x = [float(value) for value in x]

for index in range(len(fileStems)):
    endResult = endResultList[index]
    plt.plot(
        x,
        endResult.beforePruningTestError,
        marker="o",
        linestyle="dashed",
        color=colors[index],
        label=fileStems[index] + " before pruning",
    )
    plt.plot(
        x,
        endResult.afterPruningTestError,
        marker="o",
        color=colors[index],
        label=fileStems[index] + " after pruning",
    )
    plt.plot(
        x,
        idealResult[0],
        marker="o",
        color=colors[index],
        linestyle="dotted",
        label=fileStems[index] + " ideal classifier",
    )

plt.title("Datasets: " + str(fileStems))
plt.xticks(x, x)
plt.xlabel("C value used for data set generation")
plt.ylabel("Percentage error")
plt.legend()
plt.savefig(fname="errorGraph")

# Cleanup.
for fileStem in fileStems:
    subprocess.run(["rm"] + glob.glob(fileStem + ".*"))
