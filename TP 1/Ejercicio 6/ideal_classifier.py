from math import sqrt


def pointDistance(pointA, pointB):
    return sqrt(
        sum([(pointA[index] - pointB[index]) ** 2 for index in range(len(pointA))])
    )


centerList = {
    "diagonal.test": [[1] * 5, [-1] * 5],
    "paralelo.test": [[1] + [0] * 4, [-1] + [0] * 4],
}


def run_ideal_classifier(fileName: str):
    errorCount = 0
    lineCount = 0
    inputFile = open(fileName, "r")
    for line in inputFile.readlines():
        lineCount += 1
        coordsAndKlass = list(map(float, line.split(",")))
        klass = coordsAndKlass[-1]
        coords = coordsAndKlass[:-1]
        distList = [pointDistance(coords, center) for center in centerList[fileName]]
        guessedKlass = distList.index(min(distList))
        if guessedKlass != klass:
            errorCount += 1
    # No se en que momento lo di vuelta... pero soy demasiado bueno adivinando mal.
    percentageError = (1 - errorCount / lineCount) * 100
    # print("errorCount: " + str(errorCount) + "    lineCount: " + str(lineCount))
    # print("percentageError: " + str(percentageError))
    return percentageError
