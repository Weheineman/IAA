#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess

fileStem = sys.argv[1]
inputFile = open(fileStem + ".mse", "r")

clasifTrainErr = []
clasifValidationErr = []
clasifTestErr = []
epochCount = [0]

epochSaveCount = 10

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
    clasifTrainErr.append(clasifTrainErrValue)
    clasifValidationErr.append(clasifValidationErrValue)
    clasifTestErr.append(clasifTestErrValue)
    epochCount.append(epochCount[-1] + epochSaveCount)

# Plot generated data.
x = epochCount[1:]
# The first x datapoint will be epochSaveCount * (start + 1)
start = 9
step = 10

plt.plot(
    x[start::step],
    clasifTrainErr[start::step],
    marker="o",
    color="tab:purple",
    label="Classification Train Error",
)
plt.plot(
    x[start::step],
    clasifValidationErr[start::step],
    marker="o",
    color="tab:brown",
    label="Classification Validation Error",
)
plt.plot(
    x[start::step],
    clasifTestErr[start::step],
    marker="o",
    color="tab:gray",
    label="Classification Test Error",
)


plt.title(f"Dataset: {fileStem}")
plt.xticks(x[start::step], x[start::step])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.savefig(fname=f"{fileStem}ClasifGraph")
