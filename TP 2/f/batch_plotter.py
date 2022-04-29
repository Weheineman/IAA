#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import subprocess

fileStem = sys.argv[1]
batchSizeList = ["1", "5", "10"]
epochCount = [0]

epochSaveCount = 200
trainMseList = []
testMseList = []

for batchSize in batchSizeList:
    inputFile = open(f"{fileStem}_{batchSize}.mse", "r")
    epochCount = [0]
    trainMse = []
    validationMse = []
    testMse = []
    for line in inputFile.readlines():
        [
            stochasticMseValue,
            trainMseValue,
            validationMseValue,
            testMseValue,
            clasifTrainErrValue,
            clasifValidationErrValue,
            clasifTestErrValue,
        ] = map(float, line.split())
        trainMse.append(trainMseValue)
        testMse.append(testMseValue)
        epochCount.append(epochCount[-1] + epochSaveCount)
    trainMseList.append(trainMse)
    testMseList.append(testMse)

# Plot generated data.
x = epochCount[1:]
# The first x datapoint will be epochSaveCount * (start + 1)
start = 49
step = 50

colors = ["tab:blue", "tab:green", "tab:red"]

for index in range(len(batchSizeList)):
    batchSize = batchSizeList[index]
    plt.plot(
        x[start::step],
        trainMseList[index][start::step],
        marker="o",
        color=colors[index],
        label=f"Batch size {batchSize}",
    )
plt.title(f"Sunspots Training Error")
plt.xticks(x[start::step], x[start::step])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.savefig(fname=f"{fileStem}TrainGraph")

plt.clf()

for index in range(len(batchSizeList)):
    batchSize = batchSizeList[index]
    plt.plot(
        x[start::step],
        testMseList[index][start::step],
        marker="o",
        color=colors[index],
        label=f"Batch size {batchSize}",
    )
plt.title(f"Sunspots Test Error")
plt.xticks(x[start::step], x[start::step])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.savefig(fname=f"{fileStem}TestGraph")
