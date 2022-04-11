#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import subprocess

fileStem = sys.argv[1]
if("--run" in sys.argv):
    subprocess.run(["bp", fileStem])
inputFile = open(fileStem + ".mse", "r")

stochasticMse = []
trainMse = []
validationMse = []
testMse = []
clasifTrainErr = []
clasifValidationErr = []
clasifTestErr = []
epochCount = [0]

epochSaveCount = 1

for line in inputFile.readlines():
    [stochasticMseValue, trainMseValue, validationMseValue, testMseValue, clasifTrainErrValue, clasifValidationErrValue, clasifTestErrValue] = map(float, line.split())
    stochasticMse.append(stochasticMseValue)
    trainMse.append(trainMseValue)
    validationMse.append(validationMseValue)
    testMse.append(testMseValue)
    clasifTrainErr.append(clasifTrainErrValue)
    clasifValidationErr.append(clasifValidationErrValue)
    clasifTestErr.append(clasifTestErrValue)
    epochCount.append(epochCount[-1] + epochSaveCount)

# Plot generated data.
x = epochCount[1:]
step = 50

plt.plot(
    x[::step],
    stochasticMse[::step],
    marker="o",
    color="tab:blue",
    label="Stochastic MSE",
)
plt.plot(
    x[::step],
    trainMse[::step],
    marker="o",
    color="tab:orange",
    label="Training MSE",
)
plt.plot(
    x[::step],
    validationMse[::step],
    marker="o",
    color="tab:green",
    label="Validation MSE",
)
plt.plot(
    x[::step],
    testMse[::step],
    marker="o",
    color="tab:red",
    label="Test MSE",
)

plt.plot(
    x[::step],
    clasifTestErr[::step],
    marker="o",
    color="tab:purple",
    label="Classification Test Error",
)
plt.plot(
    x[::step],
    clasifValidationErr[::step],
    marker="o",
    color="tab:brown",
    label="Classification Validation Error",
)
plt.plot(
    x[::step],
    clasifTestErr[::step],
    marker="o",
    color="tab:gray",
    label="Classification Test Error",
)

plt.title("Dataset: " + fileStem)
plt.xticks(x[::step], [val - 1 for val in x[::step] ])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.show()
# plt.savefig(fname="errorGraph")

# Cleanup.
if("--clean" in sys.argv):
    subprocess.run(["rm"] + glob.glob("*.wts"))
    subprocess.run(["rm", fileStem + ".predic"])
    subprocess.run(["rm", fileStem + ".mse"])
