#!/usr/bin/python

import subprocess
import sys

fileStem = sys.argv[1]
klassCount = int(sys.argv[2])

subprocess.run(["rm", f"{fileStem}.mse"])

# Vectorize classes using 0.9 for the class index and 0.1 for all others.
for extension in ["data", "test"]:
    inputFile = open(f"{fileStem}_original.{extension}", "r")
    inputLines = inputFile.readlines()
    outputLines = []
    for line in inputLines:
        floatList = [float(value) for value in line.split(",")]
        klass = int(floatList[-1])
        klassVector = [0.1] * klassCount
        klassVector[klass] = 0.9
        outputLines.append(",".join(map(str, floatList[:-1] + klassVector)) + "\n")

    inputFile = open(f"{fileStem}.{extension}", "w")
    inputFile.writelines(outputLines)
    inputFile.close()

subprocess.run(["bp_multiclass", fileStem])

# Normalize predictions from vectors to integers.
predicFile = open(f"{fileStem}.predic", "r")
inputLines = predicFile.readlines()
outputLines = []
for line in inputLines:
    floatList = [float(value) for value in line.split("\t")[:-1]]
    klassVector = floatList[-klassCount:]
    klass = klassVector.index(max(klassVector))
    outputLines.append(",".join(map(str, floatList[:-klassCount] + [klass])) + "\n")

predicFile = open(f"{fileStem}_original.predic", "w")
predicFile.writelines(outputLines)
predicFile.close()
