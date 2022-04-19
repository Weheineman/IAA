"""
k_nn_regression.py : Regresión con k-primeros vecinos.

GGW
"""

from datetime import datetime
from statistics import mean
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import sys


class InputData:
    def __init__(self, file_stem: str):
        self.file_stem = file_stem

        config_file = open(f"{file_stem}.knn", "r")
        self.k = int(config_file.readline())
        self.n_features = int(config_file.readline())
        self.n_points = int(config_file.readline())
        self.n_train = int(config_file.readline())
        self.n_valid = self.n_points - self.n_train
        self.n_test = int(config_file.readline())
        self.seed = int(config_file.readline())
        self.verbosity = int(config_file.readline())

        # Chequear semilla para la funcion rand().
        if self.seed == 0:
            self.seed = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        random.seed(self.seed)

    def print_config(self) -> None:
        print("\nK Nearest Neighbors:")
        print(f"K: {self.k}")
        print(f"Cantidad de entradas: {self.n_features}")
        print(f"Archivo de patrones: {self.file_stem}.data")
        print(f"Cantidad total de patrones: {self.n_points}")
        print(f"Cantidad de patrones de entrenamiento: {self.n_train}")
        print(f"Cantidad de patrones de validacion: {self.n_valid}")
        print(f"Cantidad de patrones de test: {self.n_test}")
        print(f"Semilla para la funcion rand(): {self.seed}")

    def read_data(self) -> None:
        col_names = [f"x{n}" for n in range(self.n_features)] + ["target"]

        # Leer datos de entrenamiento.
        self.train_df = pd.read_csv(f"{self.file_stem}.data", names=col_names)

        # Separar datos de validacion.
        self.train_df, self.valid_df = train_test_split(
            self.train_df, test_size=self.n_valid
        )

        # Leer datos de test.
        self.test_df = pd.read_csv(f"{self.file_stem}.test", names=col_names)


class KNearestNeighborsReg:
    def __init__(self, input: InputData):
        self.point_df = input.train_df
        self.n_features = input.n_features
        self.k_list = [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 100]
        self._optimize_k(input.valid_df)

    # Dada una lista de coordenadas predice el target value.
    def predict_target(self, query: pd.DataFrame) -> float:
        # Calcula la distancia de la query a todos los puntos de entrenamiento.
        self.point_df["query_dist"] = np.sqrt(
            ((self.point_df.iloc[:, : self.n_features] - query) ** 2).sum(axis=1)
        )

        # Ordena por distancia y hace la regresión con la ecuación
        # vista en clase.
        sorted_points = self.point_df.sort_values("query_dist")
        # Quito puntos a distancia 0.
        sorted_points = sorted_points.loc[sorted_points["query_dist"] > 0]
        k_neighbors = sorted_points.iloc[:self.k]
        k_neighbors["w"] = k_neighbors["query_dist"]**(-2)
        return (k_neighbors["w"] * k_neighbors["target"]).sum() / k_neighbors["w"].sum()

    # Predice el target value de una lista de puntos.
    # Devuelve el mse.
    # Si out_file es no vacio, escribe el archivo con las predicciones.
    def predict_case_list(self, case_list: pd.DataFrame, out_file: str = "") -> float:
        case_list["predic"] = case_list.iloc[:, : self.n_features].apply(
            self.predict_target, axis=1
        )
    
        error = mean_squared_error(case_list["target"],case_list["predic"])

        if len(out_file):
            case_list.to_csv(out_file)

        return error

    # Predice sobre conjuntos de train, validation y test.
    # Devuelve el error de cada uno por consola.
    # Devuelve un archivo .predic con las predicciones de test.
    def predict(self, input: InputData):
        train_err = self.predict_case_list(input.train_df)

        if input.n_valid:
            val_err = self.predict_case_list(input.valid_df)
        else:
            val_err = train_err

        if input.n_test:
            test_err = self.predict_case_list(input.test_df)
        else:
            test_err = 0

        print(f"k óptimo: {self.k}")
        print(f"Error de entrenamiento: {train_err*100}%")
        print(f"Error de validación: {val_err*100}%")
        print(f"Error de test: {test_err*100}%")

    # Elije el k de k_list que minimiza el error de predicción en valid_df.
    def _optimize_k(self, valid_df: pd.DataFrame):
        best_k = None
        best_err = None

        for current_k in self.k_list:
            self.k = current_k
            current_err = self.predict_case_list(valid_df)
            if best_err is None or current_err < best_err:
                best_err = current_err
                best_k = current_k

        self.k = best_k


def main():
    if len(sys.argv) != 2:
        print("Modo de uso: python k_nn.py <filename>")
        print("donde filename es el nombre del archivo (sin extension)")
        quit()

    file_stem = sys.argv[1]
    input_data = InputData(file_stem)
    # input_data.print_config()
    input_data.read_data()
    k_nn = KNearestNeighborsReg(input_data)
    k_nn.predict(input_data)


if __name__ == "__main__":
    main()
