#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import subprocess

fileStem = sys.argv[1]
inputFile = open(fileStem + ".mse", "r")

trainMse = []
validationMse = []
testMse = []
epochCount = [0]

epochSaveCount = 200

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
    validationMse.append(validationMseValue)
    testMse.append(testMseValue)
    epochCount.append(epochCount[-1] + epochSaveCount)

# Plot generated data.
x = epochCount[1:]
# The first x datapoint will be epochSaveCount * (start + 1)
start = 49
step = 50


plt.plot(
    x[start::step],
    trainMse[start::step],
    marker="o",
    color="tab:orange",
    label="Training MSE",
)
plt.plot(
    x[start::step],
    validationMse[start::step],
    marker="o",
    color="tab:green",
    label="Penalization",
)
plt.plot(
    x[start::step],
    testMse[start::step],
    marker="o",
    color="tab:red",
    label="Test MSE",
)


plt.title(f"Dataset: {fileStem[:-2]}\nGamma: 10^-{fileStem[-1]}")
plt.xticks(x[start::step], x[start::step])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.savefig(fname=f"{fileStem}MSEGraph")
