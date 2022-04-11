#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import subprocess

fileStem = 'dos_elipses'
if("--run" in sys.argv):
    subprocess.run(["rm"] + glob.glob("*.wts"))
    subprocess.run(["rm", fileStem + ".predic"])
    subprocess.run(["rm", fileStem + ".mse"])
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

epochSaveCount = 200

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
# The first x datapoint will be epochSaveCount * (start + 1)
start = 9
step = 19

# plt.plot(
#     x[start::step],
#     stochasticMse[start::step],
#     marker="o",
#     color="tab:blue",
#     label="Stochastic MSE",
# )
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
    label="Validation MSE",
)
plt.plot(
    x[start::step],
    testMse[start::step],
    marker="o",
    color="tab:red",
    label="Test MSE",
)

# plt.plot(
#     x[start::step],
#     clasifTestErr[start::step],
#     marker="o",
#     color="tab:purple",
#     label="Classification Test Error",
# )
# plt.plot(
#     x[start::step],
#     clasifValidationErr[start::step],
#     marker="o",
#     color="tab:brown",
#     label="Classification Validation Error",
# )
# plt.plot(
#     x[start::step],
#     clasifTestErr[start::step],
#     marker="o",
#     color="tab:gray",
#     label="Classification Test Error",
# )

learningRate = 0.001
momentum = 0.9

plt.title(f"Dataset: {fileStem}\n Learning rate: {learningRate}  Momentum: {momentum}")
plt.xticks(x[start::step], x[start::step])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.savefig(fname=f"errorGraph_rate{learningRate}_momentum{momentum}.png")
