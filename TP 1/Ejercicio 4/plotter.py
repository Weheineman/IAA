#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

inputFile = open(sys.argv[1], "r")
x = []
y = []
klassList = []

for line in inputFile.readlines():
    [xCoord, yCoord, klass] = map(float, line.split("\t"))
    x.append(xCoord)
    y.append(yCoord)
    klassList.append(klass)


plt.scatter(x, y, c=klassList)
plt.show()
