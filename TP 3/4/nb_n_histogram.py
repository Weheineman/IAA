"""
nb_n.py : Clasificador Naive Bayes usando la aproximacion de funciones normales
para features continuos.
Formato de datos: c4.5
La clase a predecir tiene que ser un numero comenzando de 0: por ejemplo,
para 3 clases, las clases deben ser 0, 1, 2.

GGW - Ultima revision: 13/10/2021
"""

from datetime import datetime
from numpy import log
from typing import List
import random
import sys


class Histogram:
    def __init__(self, data: List[float], binCount: int):
        self.binCount = binCount

        # divisions son los valores que separan los bins.
        minValue = min(data)
        maxValue = max(data)
        if binCount > 1:
            step = (maxValue - minValue) / binCount
            self.divisions = [minValue] * (binCount - 1)
            for index in range(binCount - 1):
                self.divisions[index] += step * (index + 1)

        # Probabilidad por cada bin usando m-estimate of probability.
        # 6.9.1.1, pág. 179 del libro de Mitchell (con p=1/N_bins y m=1).
        m = 1
        p = 1 / self.binCount
        self.binProbability = [m * p for x in range(binCount)]
        for value in data:
            self.binProbability[self.calculate_bin(value)] += 1
        self.binProbability = [value / (len(data) + m) for value in self.binProbability]

    def calculate_bin(self, value: float) -> int:
        if self.binCount == 1:
            return 0
        # El bin que le corresponde a un valor es igual a la cantidad de divisiones
        # a las que es mayor.
        return sum([int(value > division) for division in self.divisions])

    def probability(self, value: float) -> float:
        return self.binProbability[self.calculate_bin(value)]


class InputData:
    def __init__(self, fileStem: str):
        self.fileStem = fileStem

        configFile = open(f"{fileStem}.nb", "r")
        self.inputCount = int(configFile.readline())
        self.classCount = int(configFile.readline())
        self.binCount = int(configFile.readline())
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
        print(f"Cantidad de entradas: {self.inputCount}")
        print(f"Cantidad de clases: {self.classCount}")
        print(f"Cantidad de bins (dado por el input): {self.binCount}")
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
            klass = int(line[-1])
            values = [float(x) for x in line[:-1]]
            self.trainValues[klass].append(values)

        # Leer datos de validacion.
        for strLine in lines[self.trainCount :]:
            line = strLine.split(",")
            klass = int(line[-1])
            values = [float(x) for x in line[:-1]]
            self.validValues[klass].append(values)

        # Leer datos de test.
        testFile = open(f"{self.fileStem}.test", "r")
        for strLine in testFile.readlines():
            line = strLine.split(",")
            klass = int(line[-1])
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
            for klass in range(input.classCount)
        ]

        self.binCount = input.binCount
        self.train(input)
        # Esta version no optimiza bins, usa el valor del input.
        # self.optimize_bins(input)

    # Optimizar la cantidad de bins, si hay conjunto de validacion.
    # Prueba todas las cantidades de bin en el intervalo [1, maxBins).
    def optimize_bins(self, input: InputData, maxBins: int = 200) -> None:
        if input.validCount:
            validError = self.predict_class_case_list(input.validValues)
            bestBinCount = self.binCount
            for currentBins in range(1, maxBins):
                self.binCount = currentBins
                self.train(input)
                currentError = self.predict_class_case_list(input.validValues)
                # Si el resultado es mejor que mi mejor resultado, lo actualiza.
                if currentError < validError:
                    validError = currentError
                    bestBinCount = currentBins
            # Se reentrena el modelo con la cantidad de bins optima.
            self.binCount = bestBinCount
            self.train(input)

    # Entrenar el modelo para una cantidad de bins dada.
    def train(self, input: InputData) -> None:
        # Histograma para cada par (clase, atributo).
        self.distribution = [
            [
                Histogram(input.get_train_values(klass, attribute), self.binCount)
                for attribute in range(input.inputCount)
            ]
            for klass in range(input.classCount)
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
        # print(f"Cantidad de bins (luego de optimizar): {self.binCount}")
        # print("Errores:")
        # print(f"Entrenamiento: {trainError * 100}%")
        # print(f"Validacion: {validError * 100}%")
        # print(f"Test: {testError * 100}%")

        # Output en el formato que espera generate_errors.py.
        print(f"{trainError*100} {validError*100} {testError*100}")


def main():
    if len(sys.argv) != 2:
        print("Modo de uso: nb_n_histogram.py <filename>")
        print("donde filename es el nombre del archivo (sin extension)")
        quit()

    fileStem = sys.argv[1]
    inputData = InputData(fileStem)
    inputData.read_data()
    # inputData.print_config()
    naiveBayes = NaiveBayes(inputData)
    naiveBayes.predict(inputData)


if __name__ == "__main__":
    main()
