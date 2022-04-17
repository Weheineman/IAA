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
avgFile = open("NBAvgError.txt", "w")
medianFile = open("NBMedianError.txt", "w")

# Generate errors.
for fileStem in fileStems:
    for d in dimensionList:
        # Generate points.
        subprocess.run(["./" + fileStem, d, testSize, C])
        subprocess.run(["mv", fileStem + ".data", fileStem + ".test"])

        # Change dimension in .nb
        nbFile = open(f"{fileStem}.nb", "r")
        lineList = nbFile.readlines()
        lineList[0] = f"{d}\n"

        nbFile = open(f"{fileStem}.nb", "w")
        nbFile.writelines(lineList)
        nbFile.close()

        trainErrorList = []
        testErrorList = []

        # Generate Naive Bayes error.
        for iteration in range(iterationCount):
            subprocess.run(["./" + fileStem, d, trainingSize, C])
            completed = subprocess.run(
                ["python", "nb_n.py", fileStem], capture_output=True, text=True
            )
            print(f"fileStem: {fileStem}, d: {d}")
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
cleanupExtensions = [".names", ".data", ".test", ".predic"]
for fileStem in fileStems:
    for extension in cleanupExtensions:
        subprocess.run(["rm", fileStem + extension])
