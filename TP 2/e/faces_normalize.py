#!/usr/bin/python

import subprocess
import sys

fileStem = "faces"

# Normalize pixel values linearly from range [0, 255] to range [0,1].
for extension in ["data", "test"]:
    inputFile = open(f"{fileStem}_not_normalized.{extension}", "r")
    inputLines = inputFile.readlines()
    outputLines = []
    for line in inputLines:
        floatList = [float(value) / 255 for value in line.split(",")[:-1]]
        outputLines.append(",".join(map(str, floatList)) + f",{line.split(',')[-1]}")

    inputFile = open(f"{fileStem}_original.{extension}", "w")
    inputFile.writelines(outputLines)
    inputFile.close()
