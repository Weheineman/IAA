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
* [`GaussianNB` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB) porque arrojaba resultados parecidos a mi Naive Bayes gaussiano y tiene una linda interfaz.


Usé una modificación de `err_estimator.py`, que programé para el TP de Ensembles de TMD. El script estima el error usando 10-fold cross validation de la siguiente forma:

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

Ocurre algo similar al caso anterior: no siento que la segunda corrida aporte valor significativo. Los valores elegidos fueron 1 casi siempre para `degree` y caos total para `C`. Pareciera que no se obtiene una ventaja por sobre el SVM lineal. En el ejercicio siguiente veremos si es "verdad".

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

Una corrida de un algoritmo sencillo y feliz (y rápido!) le ganó a varias corridas lentas donde intentaba ajustar parámetros. Me sorprende y la verdad me pone un poco triste (por mi esfuerzo en vano). Aunque el próximo ejercicio dirá "en realidad" si es mejor.

## Naive Bayes
Intenté usar `nb_n.py`, copiado del TP de Naive Bayes y me pregunté por qué no aprendí antes a usar `pandas`. Luego de renegar un rato probé con [`GaussianNB` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB) y para corridas simples (haciendo random shuffle y luego usando 4/5 de los datos para entrenar y 1/5 para validar) los errores eran muy similares (ambos valores rondaban 0.55 en promedio). Por lo que decidí dejar de sufrir y usar `GaussianNB`.

Parámetros de `err_estimator.py`:
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

# Ejercicio b
Hice un script `t_test.py`, una modificación de `err_estimator.py`.

## Decision Tree vs. Naive Bayes
Parámetros de `t_test.py`:
```python
file_stem = "BBBs"
estimator_a = GaussianNB()
estimator_b = DecisionTreeClassifier(criterion="entropy")
n_splits = 10
table_value = 2.26 # t_{N, v} for n=95% and v=9
param_grid_a = {}
param_grid_b = {}
```

```
Delta (mean): 0.2765389082462254
Delta error: 0.060841796948349076
```

Si entendí bien lo que dice Mitchell, con 95% de confidencia podemos afirmar que la diferencia de error de test entre Decision Tree y Naive Bayes está en el intervalo (aproximado) `[0.270, 0.282]`. O sea que hay una diferencia grandísima (lo cual uno intuye al ver los números en el ejercicio anterior).

## Decision Tree vs. SVM Lineal

Parámetros de `t_test.py`:
```python
file_stem = "BBBs"
estimator_a = SVC(kernel="poly", degree=1)
estimator_b = DecisionTreeClassifier(criterion="entropy")
n_splits = 10
table_value = 2.26 # t_{N, v} for n=95% and v=9
param_grid_a = {"C": np.arange(3, 40, 0.5)}
param_grid_b = {}
```

```
Delta (mean): 0.08403019744483158
Delta error: 0.05031468153731031
```

¡Da positivo de nuevo! Pero con una diferencia de error esperado mucho más chica. Tiene sentido. Siento mi tristeza del apartado a justificada con un 95% de confidencia.

## SVM Lineal vs. SVM Polinomial
Quiero que un t-test me de negativo. A ver si sale.

Parámetros de `t_test.py`:
```python
file_stem = "BBBs"
estimator_a = SVC(kernel="poly")
estimator_b = SVC(kernel="poly", degree=1)
n_splits = 10
table_value = 2.26 # t_{N, v} for n=95% and v=9
param_grid_a = {"degree": range(1, 5), "C": np.arange(0.5, 10, 0.1)}
param_grid_b = {"C": np.arange(3, 40, 0.5)}
```

```
Delta (mean): 0.016724738675958185
Delta error: 0.0227695964446012
```

Dio negativo, qué bueno. Hasta ahora todos los resultados se corresponden con lo que uno (al menos yo) intuye al ver la tabla del ejercicio anterior.

# Ejercicio c
Elegí [`digits` de scikit](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits) porque sólo usamos ANN para reconocimiento de imágenes y quería probar otros métodos. Además me queda cómodo usar algo de sklearn.

Usé copias modificadas de los scripts de los ejercicios anteriores.

## SVM Lineal
Parámetros de la primera corrida de `err_estimator.py`:
```python 
estimator = SVC(kernel="poly", degree=1)
n_splits = 10
param_grid = {"C": np.logspace(-6, 3, 20)}
```

```
Estimated test error (mean): 0.01669770328988208
Standard deviation: 0.006108650857090372
```

Parámetros de la segunda corrida de `err_estimator.py`:
```python 
estimator = SVC(kernel="poly", degree=1)
n_splits = 10
param_grid = {"C": np.arange(1.4, 5, 0.05)}
```

```
Estimated test error (mean): 0.016700806952203596
Standard deviation: 0.008989006076555485
```

Otra vez sopa. Al menos esta vez anduvo mucho mejor.

## Decision Tree
Parámetros de `err_estimator.py`:
```python 
estimator = DecisionTreeClassifier(criterion="entropy")
n_splits = 10
param_grid = {}
```

```
Estimated test error (mean): 0.13024518932340162
Standard deviation: 0.021887858705406583
```

Al fin se cumple mi predicción mental de que algo tan sencillo como un árbol de decisión no puede ganarle consistentemente a SVM. Empezaba a pensar seriamente si la implementación de scikit estaba tuneada de alguna forma extraña (leí la documentación y me parece que no).

## Naive Bayes
Estimo que le va a ir muy mal porque los píxeles dependen mucho entre sí.

Parámetros de `err_estimator.py`:
```python 
estimator = GaussianNB()
n_splits = 10
param_grid = {}
```

```
Estimated test error (mean): 0.1652855369335816
Standard deviation: 0.016021087166845537
```

Es un orden de magnitud peor que SVM pero aún así mucho mejor de lo que esperaba. De hecho lo esperaba notablemente peor que Decision Tree... será peor con 95% de confidencia? Veamos.

## Decision Tree vs. Naive Bayes
Parámetros de `t_test.py`:
```python
estimator_a = GaussianNB()
estimator_b = DecisionTreeClassifier()
n_splits = 10
table_value = 2.26 # t_{N, v} for n=95% and v=9
param_grid_a = {}
param_grid_b = {}
```

```
Delta (mean): 0.028448168839230282
Delta error: 0.02455944019562996
```

Por poco, pero da positivo! De esto además concluimos que también hay diferencia significativa entre SVM y los otros dos métodos... pero eso era obvio a primera vista.