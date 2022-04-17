#!/usr/bin/python

"""
nb_n.py : Clasificador Naive Bayes usando la aproximacion de funciones normales
para features continuos.
Formato de datos: c4.5
La clase a predecir tiene que ser un numero comenzando de 0: por ejemplo,
para 3 clases, las clases deben ser 0, 1, 2.

GGW - Ultima revision: 13/10/2021
"""

from datetime import datetime
from numpy import exp, log, mean, pi, sqrt, var
from typing import List
import random
import sys


class NormalDistribution:
    def __init__(self, data: List[float]):
        # Toda probabilidad vale al menos minProbability, para evitar ceros.
        self.minProbability = sys.float_info.epsilon

        # Es posible que no haya valores para una clase.
        self.emptyDistribution = len(data) == 0

        if not self.emptyDistribution:
            self.mean = mean(data)
            self.variance = var(data)

    def probability(self, x: float) -> float:
        if self.emptyDistribution:
            return self.minProbability
        else:
            exponent = -(x - self.mean) * (x - self.mean) / (2 * self.variance)
            denominator = sqrt(2 * pi * self.variance)
            return max(self.minProbability, exp(exponent) / denominator)


class InputData:
    def __init__(self, fileStem: str):
        self.fileStem = fileStem

        configFile = open(f"{fileStem}.nb", "r")
        self.inputCount = int(configFile.readline())
        self.classCount = int(configFile.readline())
        self.patternCount = int(configFile.readline())
        self.trainCount = int(configFile.readline())
        self.validCount = self.patternCount - self.trainCount
        self.testCount = int(configFile.readline())
        self.seed = int(configFile.readline())
        self.verbosity = int(configFile.readline())

        # Chequear semilla para la funcion rand().
        if self.seed == 0:
            self.seed = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        random.seed(self.seed)

        # Inicializar las clases como listas vacias.
        self.trainValues = [[] for x in range(self.classCount)]
        self.validValues = [[] for x in range(self.classCount)]
        self.testValues = [[] for x in range(self.classCount)]

    def print_config(self) -> None:
        print("\nNaive Bayes con distribuciones normales:")
        print(f"Cantidad de entradas:{self.inputCount}")
        print(f"Cantidad de clases:{self.classCount}")
        print(f"Archivo de patrones: {self.fileStem}")
        print(f"Cantidad total de patrones: {self.patternCount}")
        print(f"Cantidad de patrones de entrenamiento: {self.trainCount}")
        print(f"Cantidad de patrones de validacion: {self.validCount}")
        print(f"Cantidad de patrones de test: {self.testCount}")
        print(f"Semilla para la funcion rand(): {self.seed}")

    def read_data(self) -> None:
        dataFile = open(f"{self.fileStem}.data", "r")
        lines = dataFile.readlines()
        if self.seed != -1:
            random.shuffle(lines)

        # Leer datos de entrenamiento.
        for strLine in lines[: self.trainCount]:
            line = strLine.split(",")
            klass = int(line[-1][:-2])
            values = [float(x) for x in line[:-1]]
            self.trainValues[klass].append(values)

        # Leer datos de validacion.
        for strLine in lines[self.trainCount : self.patternCount]:
            line = strLine.split(",")
            klass = int(line[-1][:-2])
            values = [float(x) for x in line[:-1]]
            self.validValues[klass].append(values)

        # Leer datos de test.
        testFile = open(f"{self.fileStem}.test", "r")
        for strLine in testFile.readlines():
            line = strLine.split(",")
            klass = int(line[-1][:-2])
            values = [float(x) for x in line[:-1]]
            self.testValues[klass].append(values)

    def get_train_values(self, klass: int, parameter: int) -> List[float]:
        return [values[parameter] for values in self.trainValues[klass]]

    def train_class_size(self, klass: int) -> int:
        return len(self.trainValues[klass])


class NaiveBayes:
    def __init__(self, input: InputData):
        # Toda probabilidad vale al menos minProbability, para evitar ceros.
        self.minProbability = sys.float_info.epsilon
        self.classCount = input.classCount

        # Probabilidad de cada clase.
        self.classProbability = [
            max(self.minProbability, input.train_class_size(klass) / input.trainCount)
            for klass in range(self.classCount)
        ]

        # Distribucion normal para cada par (clase, atributo).
        self.distribution = [
            [
                NormalDistribution(input.get_train_values(klass, attribute))
                for attribute in range(input.inputCount)
            ]
            for klass in range(self.classCount)
        ]

    # Calcula el logaritmo de la probabilidad de que un vector de atributos
    # pertenezca a una clase dada, sin considerar la evidencia.
    def log_class_probability(
        self, klass: int, attributeList: List[float]
    ) -> List[float]:
        likelyhoodList = [
            self.distribution[klass][attribute].probability(attributeList[attribute])
            for attribute in range(len(attributeList))
        ]
        prior = self.classProbability[klass]
        return log(prior) + sum([log(likelyhood) for likelyhood in likelyhoodList])

    # Dado un vector de atributos devuelve la clase predicha.
    def predicted_class(self, attributeList: List[float]) -> int:
        probabilityList = [
            self.log_class_probability(klass, attributeList)
            for klass in range(self.classCount)
        ]
        return probabilityList.index(max(probabilityList))

    # Predice las clases de una lista de listas de casos (donde cada lista de
    # casos se corresponde con una clase).
    # Devuelve el mse.
    # Si outputFile es no vacio, escribe el archivo con las predicciones.
    def predict_class_case_list(
        self, classCaseList: List[List[List[float]]], outputFile: str = ""
    ) -> float:
        clasifError = 0.0
        if len(outputFile):
            predicFile = open(outputFile, "w")

        for klass in range(self.classCount):
            for case in classCaseList[klass]:
                predictedClass = self.predicted_class(case)
                clasifError += predictedClass != klass
                if len(outputFile):
                    caseStr = ", ".join([str(i) for i in case])
                    predicFile.write(f"{caseStr}, {predictedClass}\n")

        clasifError /= sum([len(caseList) for caseList in classCaseList])
        return clasifError

    # Predice sobre conjuntos de train, validation y test.
    # Devuelve el error de cada uno por consola.
    # Devuelve un archivo .predic con las predicciones de test.
    def predict(self, input: InputData):
        trainError = self.predict_class_case_list(input.trainValues)

        if input.validCount:
            validError = self.predict_class_case_list(input.validValues)
        else:
            validError = trainError

        if input.testCount:
            testError = self.predict_class_case_list(
                input.testValues, f"{input.fileStem}.predic"
            )
        else:
            testError = 0

        # print("Fin del entrenamiento.\n")
        # print("Errores:")
        # print(f"Entrenamiento:{trainError * 100}%")
        # print(f"Validacion:{validError * 100}%")
        # print(f"Test:{testError * 100}%")

        # Output en el formato que espera generate_errors.py.
        print(f"{trainError*100} {testError*100}")


def main():
    if len(sys.argv) != 2:
        print("Modo de uso: nb_n.py <filename>")
        print("donde filename es el nombre del archivo (sin extension)")
        quit()

    fileStem = sys.argv[1]
    inputData = InputData(fileStem)
    # No imprimo la configuracion porque generate_errors.py se enoja.
    # inputData.print_config()
    inputData.read_data()
    naiveBayes = NaiveBayes(inputData)
    naiveBayes.predict(inputData)


if __name__ == "__main__":
    main()
