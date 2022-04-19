import numpy as np
import subprocess

C = "0.78"
dimensionList = ["2", "4", "8", "16", "32"]
iterationCount = 7
trainingSize = "250"
testSize = "10000"
fileStems = ["diagonal", "paralelo"]
medianFile = open("KNNMedian.err", "w")
medianFile.write("d,radius,train_err,test_err\n")

# Generate errors.
for fileStem in fileStems:
    for d in dimensionList:
        # Generate points.
        subprocess.run(["./" + fileStem, d, testSize, C])
        subprocess.run(["mv", fileStem + ".data", fileStem + ".test"])

        # Change dimension in .knn
        knnFile = open(f"{fileStem}.knn", "r")
        lineList = knnFile.readlines()
        lineList[1] = f"{d}\n"

        knnFile = open(f"{fileStem}.knn", "w")
        knnFile.writelines(lineList)
        knnFile.close()

        trainErrorList = []
        testErrorList = []
        radiusList = []

        # Generate Naive Bayes error.
        for iteration in range(iterationCount):
            subprocess.run(["./" + fileStem, d, trainingSize, C])

            completed = subprocess.run(
                ["python", "nn_radius.py", fileStem], capture_output=True, text=True
            )
            [radius, trainError, testError] = [
                float(value) for value in completed.stdout.split()
            ]
            trainErrorList.append(trainError)
            testErrorList.append(testError)
            radiusList.append(radius)
            
            print(f"fileStem: {fileStem}, d: {d}, iter: {iteration + 1}/{iterationCount}")

        # Write: dimension,radius,trainingError,testError
        trainErrorList.sort()
        testErrorList.sort()
        radius = np.mean(radiusList)

        medianIndex = iterationCount // 2
        medianFile.write(
            f"{d},{radius},{trainErrorList[medianIndex]},{testErrorList[medianIndex]}\n"
        )

# Cleanup.
cleanupExtensions = [".names", ".data", ".test"]
for fileStem in fileStems:
    for extension in cleanupExtensions:
        subprocess.run(["rm", fileStem + extension])
