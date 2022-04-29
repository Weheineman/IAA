# Setup
This project uses Python version `3.8`.

From the project root directory, on a [virtual Python environment](https://virtualenvwrapper.readthedocs.io/en/latest/) (or not, if you're feeling brave), run:
```bash
pip3 install -r requirements.txt
```

Make sure that the source directory is added to your `$PYTHONPATH` environment variable.

# Ejercicio a
Las implementaciones que usé fueron:
* [`SVC` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), con `kernel=poly` y `degree=1` como SVM lineal.
* [`SVC` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), con `kernel=poly` y `degree` variable como SVM polinomial.
* [`DecisionTreeClassifier` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontree#sklearn.tree.DecisionTreeClassifier) con `criterion = "entropy"` porque, al igual que en el TP anterior, me parece que a fines de comparar resultados es lo mismo y era más rápido para mí que reaprender cómo usar la implementación de C4.5 que nos diste. 


Usé `err_estimator.py`, que programé para el TP de Ensembles de TMD. El script estima el error usando 10-fold cross validation de la siguiente forma:

1. Separo 1/10 de los datos como conjunto de validación, el resto como training usando [`StratifiedKFold` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) para mantener las proporciones de las clases en los folds.
2. Sobre los datos de training vuelvo a hacer 10-fold cross validation para elegir parámetros óptimos utilizando [`GridSearchCV` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV).
3. Entreno el modelo usando todos los datos de training (usando los parámetros óptimos) y obtengo su error de clasificación sobre el conjunto de validación.
4. Repito los pasos 1 a 3 para cada fold y promedio los errores de clasificación. El promedio es mi error estimado de test.

Si bien el enunciado no pide hacer un cross validation para encontrar los parámetros óptimos (simplemente que usemos los óptimos para un fold) ya estaba programado así que lo dejé así. Si bien el programa muestra los parámetros que eligió en cada folding, por brevedad los voy a omitir.

## SVM Lineal
Parámetros de la primera corrida de `err_estimator.py`:
```python 
file_stem = "BBBs"
estimator = SVC(kernel="poly", degree=1)
n_splits = 10
param_grid = {"C": np.logspace(-8, 4, 30)}
```

```
Estimated test error (mean): 0.34006968641114976
Standard deviation: 0.055231457340469824
```

Parámetros de la segunda corrida de `err_estimator.py`:
```python 

file_stem = "BBBs"
estimator = SVC(kernel="poly", degree=1)
n_splits = 10
param_grid = {"C": np.arange(3, 40, 0.5)}
```

```
Estimated test error (mean): 0.3251451800232289
Standard deviation: 0.06584272431844551
```

Al igual que en TMD, me cuesta un montón optimizar parámetros en SVM. Pareciera que la segunda corrida no importa. Los `C` óptimos dan distinto en cada corrida, como simulando una distribución uniforme en todo el intervalo. Sinceramente no entiendo.

## SVM Polinomial
Parámetros de la primera corrida de `err_estimator.py`:
```python 
file_stem = "BBBs"
estimator = SVC(kernel="poly")
n_splits = 10
param_grid = {"degree": range(1, 5), "C": np.logspace(-8, 4, 30)}
```

```
Estimated test error (mean): 0.3591173054587688
Standard deviation: 0.03391643986321489
```

Parámetros de la segunda corrida de `err_estimator.py`:
```python 
file_stem = "BBBs"
estimator = SVC(kernel="poly")
n_splits = 10
param_grid = {"degree": range(1, 5), "C": np.arange(0.5, 10, 0.1)}
```

```
Estimated test error (mean): 0.3321138211382113
Standard deviation: 0.06567605360376269
```

Ocurre algo similar al caso anterior: no siento que la segunda corrida aporte valor significativo. Los valores elegidos fueron 1 casi siempre para `degree` y caos total para `C`. Pareciera que no se obtiene una ventaja por sobre el SVM lineal.

## Decision Tree
Parámetros de `err_estimator.py`:
```python 
file_stem = "BBBs"
estimator = DecisionTreeClassifier(criterion = "entropy")
n_splits = 10
param_grid = {}
```

```
Estimated test error (mean): 0.26765389082462254
Standard deviation: 0.05839598941503598
```

Una corrida de un algoritmo sencillo y feliz (y rápido!) le ganó a varias corridas lentas donde intentaba ajustar parámetros. Me sorprende y la verdad me pone un poco triste (por mi esfuerzo en vano). Aunque el próximo ejercicio dirá en realidad si es mejor.

## Naive Bayes
Intenté usar `nb_n.py`, copiado del TP de Naive Bayes y me pregunté por qué no aprendí antes a usar `pandas`. Luego de renegar un rato probé con [`GaussianNB` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB) y para corridas simples (haciendo random shuffle y luego usando 4/5 de los datos para entrenar y 1/5 para validar) los errores eran muy similares (ambos valores rondaban 0.55 en promedio). Por lo que decidí dejar de sufrir y usar `GaussianNB`.

```python 
file_stem = "BBBs"
estimator = GaussianNB()
n_splits = 10
param_grid = {}
```

```
Estimated test error (mean): 0.5367015098722415
Standard deviation: 0.07701355648395544
```

Auch. Algo me dice que los features no tienen distribución normal (o son fuertemente dependientes).

## Tabla de resultados

|                | Media   | Desviación Estándar |
|----------------|---------|---------------------|
| SVM Lineal     | 0.32514 | 0.06584             |
| SVM Polinomial | 0.33211 | 0.06567             |
| Decision Tree  | 0.26765 | 0.05839             |
| Naive Bayes    | 0.53670 | 0.07701             |
