"""
k_nn.py : Clasificador de k-primeros vecinos.

GGW
"""

from datetime import datetime
from scipy.spatial.distance import pdist, squareform
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


class NearestNeighborsRadius:
    def __init__(self, input: InputData):
        self.point_df = input.train_df
        self.n_features = input.n_features
        self._generate_radius_list(input.train_df)
        self.radius = self.radius_list[0]
        self._optimize_radius(input.valid_df)

    # Dada una lista de coordenadas predice su clase.
    def predict_class(self, query: pd.DataFrame) -> int:
        # Calcula la distancia de la query a todos los puntos de entrenamiento.
        self.point_df["query_dist"] = np.sqrt(
            ((self.point_df.iloc[:, : self.n_features] - query) ** 2).sum(axis=1)
        )

        # Ordena por distancia y se queda con las clases de los puntos a
        # distancia menor de la máxima.
        sorted_dist = self.point_df.sort_values("query_dist")
        in_radius = sorted_dist.loc[sorted_dist["query_dist"] < self.radius]

        # Si no hay nadie a la distancia elegida, toma la clase más cercana.
        if len(in_radius) == 0:
            return sorted_dist.iloc[0]["target"]

        targets_in_radius = in_radius["target"]

        # Si hay empate, quita el punto más lejano y lo vuelve a intentar.
        # Eventualmente hay un solo punto, por lo que siempre resuelve empates.
        for n_points in range(len(targets_in_radius), 0, -1):
            prediction = targets_in_radius.iloc[:n_points].mode()
            if len(prediction) == 1:
                return prediction

    # Predice las clases de una lista de puntos.
    # Devuelve el error de clasificacion.
    # Si out_file es no vacio, escribe el archivo con las predicciones.
    def predict_case_list(self, case_list: pd.DataFrame) -> float:
        case_list["predic"] = case_list.iloc[:, : self.n_features].apply(
            self.predict_class, axis=1
        )
        # El error es la cantidad de "predic" distintos de "target" sobre la cantidad de casos.
        clasifError = case_list["target"].ne(case_list["predic"]).sum() / len(case_list)

        return clasifError

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

        print(f"{self.radius} {train_err*100} {test_err*100}\n")

    def _generate_radius_list(self, train_df: pd.DataFrame):
        coords = train_df.iloc[:, : self.n_features]
        dists = squareform(pdist(coords))
        min_dist = np.amin(dists) + np.finfo(np.float32).eps  # Prevenir 0.
        max_dist = np.amax(dists)
        self.radius_list = np.logspace(np.log10(min_dist), np.log10(max_dist), 10)

    # Elije la max_dist de dist_list que minimiza el error de predicción en valid_df.
    def _optimize_radius(self, valid_df: pd.DataFrame):
        best_radius = None
        best_err = None

        for current_radius in self.radius_list:
            self.radius = current_radius
            current_err = self.predict_case_list(valid_df)
            if best_err is None or current_err < best_err:
                best_err = current_err
                best_radius = current_radius

        self.radius = best_radius


def main():
    if len(sys.argv) != 2:
        print("Modo de uso: python k_nn.py <filename>")
        print("donde filename es el nombre del archivo (sin extension)")
        quit()

    file_stem = sys.argv[1]
    input_data = InputData(file_stem)
    # input_data.print_config()
    input_data.read_data()
    k_nn = NearestNeighborsRadius(input_data)
    k_nn.predict(input_data)


if __name__ == "__main__":
    main()
