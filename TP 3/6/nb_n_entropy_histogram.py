"""
nb_n.py : Clasificador Naive Bayes usando la aproximacion de funciones normales
para features continuos.
Formato de datos: c4.5
La clase a predecir tiene que ser un numero comenzando de 0: por ejemplo,
para 3 clases, las clases deben ser 0, 1, 2.

GGW - Ultima revision: 17/04/2022
"""

from datetime import datetime
from numpy import log, log2, zeros
from typing import List
import pandas as pd
import random
import sys


# Calculates the entropy of a dataset.
def entropy(data: pd.DataFrame) -> float:
    entropy = 0

    for klass in pd.unique(data["target"]):
        proportion = len(data.loc[data["target"] == klass]) / len(data)
        entropy += -(proportion * log2(proportion))

    return entropy


# Calculates the histogram divisions for the column col using
# the paper given in the statement.
def hist_divisions(data: pd.DataFrame, col: str) -> List[float]:
    divisions = []
    min_entropy = None
    min_entropy_bound = None
    s_entropy = entropy(data)
    N = len(data)

    for value in data[col].unique():
        s1 = data.loc[data[col] <= value]
        s2 = data.loc[data[col] > value]
        class_info_entropy = len(s1) / N * entropy(s1) + len(s2) / N * entropy(s2)
        gain = s_entropy - class_info_entropy

        k = len(data["target"].unique())
        k1 = len(s1["target"].unique())
        k2 = len(s2["target"].unique())
        delta = log2(3 ** k - 2) - (k * s_entropy - k1 * entropy(s1) - k2 * entropy(s2))
        treshold = log2(N - 1) / N + delta / N

        # Keep the partition boundary that passes the treshold with the lowest entropy.
        if gain >= treshold and (
            min_entropy is None or class_info_entropy < min_entropy
        ):
            min_entropy = class_info_entropy
            min_entropy_bound = value

    # If a boundary was found, call recursively on each subset.
    if min_entropy_bound is not None:
        divisions = (
            hist_divisions(data.loc[data[col] <= min_entropy_bound], col)
            + [min_entropy_bound]
            + hist_divisions(data.loc[data[col] > min_entropy_bound], col)
        )

    return divisions


class DivisionHistogram:
    def __init__(self, data: List[float], divisions: List[float]):
        # divisions son los valores que separan los bins.
        self.divisions = divisions
        self.binCount = len(divisions) + 1

        # Probabilidad por cada bin usando m-estimate of probability.
        # 6.9.1.1, pág. 179 del libro de Mitchell (con p=1/N_bins y m=1).
        m = 1
        p = 1 / self.binCount
        self.binProbability = [m * p for _ in range(self.binCount)]
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

    def print_config(self) -> None:
        print("\nNaive Bayes con distribuciones normales:")
        print(f"Cantidad de entradas: {self.inputCount}")
        print(f"Cantidad de clases: {self.classCount}")
        print(f"Archivo de patrones: {self.fileStem}")
        print(f"Cantidad total de patrones: {self.patternCount}")
        print(f"Cantidad de patrones de entrenamiento: {self.trainCount}")
        print(f"Cantidad de patrones de validacion: {self.validCount}")
        print(f"Cantidad de patrones de test: {self.testCount}")
        print(f"Semilla para la funcion rand(): {self.seed}")

    def read_data(self) -> None:
        column_names = ["x", "y", "target"]

        self.trainValues = pd.read_csv(f"{self.fileStem}.data", names=column_names)
        self.testValues = pd.read_csv(f"{self.fileStem}.test", names=column_names)

    def get_train_values(self, klass: int, parameter: int) -> List[float]:
        return self.trainValues.loc[self.trainValues["target"] == klass].iloc[
            :, parameter
        ]

    def train_class_size(self, klass: int) -> int:
        return len(self.trainValues.loc[self.trainValues["target"] == klass])


class NaiveBayes:
    def __init__(self, input: InputData):
        # Toda probabilidad vale al menos minProbability, para evitar ceros.
        self.minProbability = sys.float_info.epsilon
        self.classCount = input.classCount
        self.featCount = input.inputCount

        # Probabilidad de cada clase.
        self.classProbability = [
            max(self.minProbability, input.train_class_size(klass) / input.trainCount)
            for klass in range(input.classCount)
        ]

        self.train(input)

    # Entrenar el modelo.
    def train(self, input: InputData) -> None:
        # Histograma para cada par (clase, atributo).
        feature_divisions = [
            hist_divisions(input.trainValues, input.trainValues.columns[feat])
            for feat in range(input.inputCount)
        ]

        self.distribution = [
            [
                DivisionHistogram(
                    input.get_train_values(klass, feature), feature_divisions[feature]
                )
                for feature in range(input.inputCount)
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

    # Predice las clases de una lista de casos.
    # Devuelve el mse.
    # Si outputFile es no vacio, escribe el archivo con las predicciones.
    def predict_class_case_list(
        self, caseList: pd.DataFrame, outputFile: str = ""
    ) -> float:
        features = caseList.iloc[:, :self.featCount]
        caseList["predic"] = [self.predicted_class(list(case)) for (_, case) in features.iterrows()]
        # El error es la cantidad de "predic" distintos de "target" sobre la cantidad de casos.
        clasifError = caseList["target"].ne(caseList["predic"]).sum() / len(caseList)
        
        if len(outputFile):
            caseList.to_csv(outputFile)

        return clasifError

    # Predice sobre conjuntos de train, validation y test.
    # Devuelve el error de cada uno por consola.
    # Devuelve un archivo .predic con las predicciones de test.
    def predict(self, input: InputData):
        trainError = self.predict_class_case_list(input.trainValues)

        if input.testCount:
            testError = self.predict_class_case_list(
                input.testValues, f"{input.fileStem}.predic"
            )
        else:
            testError = 0

        print("Fin del entrenamiento.\n")
        print("Errores:")
        print(f"Entrenamiento: {trainError * 100}%")
        print(f"Test: {testError * 100}%")

        # Output en el formato que espera generate_errors.py.
        # print(f"{trainError*100} {testError*100}")


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
