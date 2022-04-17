import numpy as np
import matplotlib.pyplot as plt
import sys

fileStem = sys.argv[1]
inputFile = open(f"{fileStem}.predic", "r")
x = []
y = []
klassList = []

for line in inputFile.readlines():
    [xCoord, yCoord, klass] = map(float, line.split(","))
    x.append(xCoord)
    y.append(yCoord)
    klassList.append(klass)


plt.scatter(x, y, c=klassList)
plt.savefig(f"{fileStem}Graph")
