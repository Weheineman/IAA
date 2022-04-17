from operator import add
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

iterationCount = 10
fileStem = sys.argv[1]
medianFile = open(f"NB{fileStem}MedianError.txt", "w")

# Generate errors.
for binCount in range(1, 201):
    trainErrorList = []
    validErrorList = []
    testErrorList = []

    # Write binCount to .nb file.
    nbFile = open(f"{fileStem}.nb", "r")
    lines = nbFile.readlines()
    lines[2] = f"{binCount}\n"
    nbFile = open(f"{fileStem}.nb", "w")
    nbFile.writelines(lines)
    nbFile.close()

    # Generate Naive Bayes error.
    for iteration in range(iterationCount):
        completed = subprocess.run(
            ["python", "nb_n_histogram.py", fileStem], capture_output=True, text=True
        )
        print(f"binCount: {binCount}")
        [trainError, validError, testError] = [
            float(value) for value in completed.stdout.split()
        ]
        trainErrorList.append(trainError)
        validErrorList.append(validError)
        testErrorList.append(testError)

    # Write: dimension trainingError testError
    trainErrorList.sort()
    validErrorList.sort()
    testErrorList.sort()
    medianIndex = iterationCount // 2
    medianFile.write(
        f"{binCount} {trainErrorList[medianIndex]} {validErrorList[medianIndex]} {testErrorList[medianIndex]}\n"
    )
