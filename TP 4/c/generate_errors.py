import subprocess
from scipy.stats import mode

C = "0.78"
dimensionList = ["2", "4", "8", "16", "32"]
iterationCount = 7
trainingSize = "250"
testSize = "10000"
fileStems = ["diagonal", "paralelo"]
medianFile_1 = open("KNNMedian_1.err", "w")
medianFile_opt = open("KNNMedian_opt.err", "w")
medianFile_1.write("d,k,train_err,test_err\n")
medianFile_opt.write("d,k,train_err,test_err\n")

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

        trainErrorList_1 = []
        testErrorList_1 = []
        trainErrorList_opt = []
        testErrorList_opt = []
        kList_opt = []

        # Generate Naive Bayes error.
        for iteration in range(iterationCount):
            subprocess.run(["./" + fileStem, d, trainingSize, C])
            
            completed = subprocess.run(
                ["python", "k_nn.py", fileStem], capture_output=True, text=True
            )
            [trainError, testError] = [
                float(value) for value in completed.stdout.split()
            ]
            trainErrorList_1.append(trainError)
            testErrorList_1.append(testError)

            completed = subprocess.run(
                ["python", "k_nn_optimize.py", fileStem], capture_output=True, text=True
            )
            [k, trainError, testError] = [
                float(value) for value in completed.stdout.split()
            ]
            trainErrorList_opt.append(trainError)
            testErrorList_opt.append(testError)
            kList_opt.append(k)
            
            print(f"fileStem: {fileStem}, d: {d}, iter: {iteration + 1}/{iterationCount}")

        # Write: dimension,k,trainingError,testError
        trainErrorList_1.sort()
        testErrorList_1.sort()
        trainErrorList_opt.sort()
        testErrorList_opt.sort()
        k, _ = mode(kList_opt)
        k = k[0]

        medianIndex = iterationCount // 2
        medianFile_1.write(
            f"{d},1,{trainErrorList_1[medianIndex]},{testErrorList_1[medianIndex]}\n"
        )
        medianFile_opt.write(
            f"{d},{k},{trainErrorList_opt[medianIndex]},{testErrorList_opt[medianIndex]}\n"
        )

# Cleanup.
cleanupExtensions = [".names", ".data", ".test"]
for fileStem in fileStems:
    for extension in cleanupExtensions:
        subprocess.run(["rm", fileStem + extension])
