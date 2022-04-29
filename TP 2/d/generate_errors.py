#!/usr/bin/python

from operator import add
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import glob
import re

C = "0.78"
dimensionList = ["2", "4", "8", "16", "32"]
iterationCount = 19
trainingSize = "250"
testSize = "10000"
fileStems = ["diagonal", "paralelo"]
avgFile = open("ANNAvgError.txt", "w")
medianFile = open("ANNMedianError.txt", "w")

# Generate errors.
for fileStem in fileStems:
    for d in dimensionList:
        # Generate points.
        subprocess.run(["./" + fileStem, d, testSize, C])
        subprocess.run(["mv", fileStem + ".data", fileStem + ".test"])

        # Change dimension in .net
        netFile = open(f"{fileStem}.net", "r")
        lineList = netFile.readlines()
        lineList[0] = f"{d}\n"

        netFile = open(f"{fileStem}.net", "w")
        netFile.writelines(lineList)
        netFile.close()

        trainErrorList = []
        testErrorList = []
        # Generate ANN error.
        for iteration in range(iterationCount):
            subprocess.run(["./" + fileStem, d, trainingSize, C])
            completed = subprocess.run(
                ["bp_discrete_err", fileStem], capture_output=True, text=True
            )
            [trainError, testError] = [
                float(value) for value in completed.stdout.split()
            ]
            trainErrorList.append(trainError)
            testErrorList.append(testError)

        # Write: dimension trainingError testError
        trainErrorList.sort()
        testErrorList.sort()
        medianIndex = iterationCount // 2
        medianFile.write(
            f"{d} {trainErrorList[medianIndex]} {testErrorList[medianIndex]}\n"
        )
        avgFile.write(
            f"{d} {sum(trainErrorList)/len(trainErrorList)} {sum(testErrorList)/len(testErrorList)}\n"
        )

# Cleanup.
cleanupExtensions = [".names", ".data", ".test", ".mse", ".predic"]
for fileStem in fileStems:
    for extension in cleanupExtensions:
        subprocess.run(["rm", fileStem + extension])
