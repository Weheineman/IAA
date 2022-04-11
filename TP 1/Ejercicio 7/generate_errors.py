#!/usr/bin/python

from operator import add
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import glob
import re

C = "0.78"
dimensionList = ["2", "4", "8", "16", "32"]
treeIterationCount = 30
trainingSize = "250"
testSize = "10000"
fileStems = ["diagonal", "paralelo"]
outputFile = open("treeError.txt", "w")

# Generate errors.
for fileStem in fileStems:
    for d in dimensionList:
        # Generate points.
        subprocess.run(["./" + fileStem, d, testSize, C])
        subprocess.run(["mv", fileStem + ".data", fileStem + ".test"])
        accumResults = [0] * 14

        # Generate decision tree error.
        for iteration in range(treeIterationCount):
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
        avgResults = [result / treeIterationCount for result in accumResults]
        # Write: dimension trainingError testError
        outputFile.write(f"{d} {avgResults[5]} {avgResults[12]}\n")

# Cleanup.
cleanupExtensions = [".tree", ".unpruned", ".names", ".prediction", ".data", ".test"]
for fileStem in fileStems:
    for extension in cleanupExtensions:
        subprocess.run(["rm",  fileStem + extension])
