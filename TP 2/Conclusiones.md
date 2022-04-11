
# Ejercicio 2
Veo que mejora con la cantidad de neuronas intermedias. Mis dudas son:
  1) Esto pasa infinitamente? Hay un momento en el que no mejora más? Recuerdo leer en el Mitchell que eventualmente para, pero no estoy seguro.
  2) Hay alguna "regla de oro" de cómo mejora el error respecto de la cantidad de neuronas intermedias?

# a
Aprendí que las ANN tienen una varianza muy grande y hay que correrlas varias veces con los mismos parámetros.

Al aumentar el learning rate, la ANN realiza saltos más grandes en el espacio de soluciones. Por lo tanto, le cuesta más converger. Se ve claramente en los gráficos con momentum 0.5: el de learning rate 0.001 converge más rápido porque si bien hace saltos más pequeños, "hace saltos más correctos" (en la gráfica es decreciente).

Si observamos las gráficas con learning rate 0.01, vemos que en la que no hay momentum, converge rápidamente. En la que tiene momentum 0.5, converge más lentamente pero a una solución mejor. Entiendo que logra salir de varios mínimos locales para encontrar un mínimo mejor. En la de momentum 0.9, vemos cómo este fenómeno está exagerado al punto en el que sale de mínimos para terminar en mínimos peores.

No me parece claro que una combinación sea necesariamente mejor que otra. Sí observo que tienen efectos distintos sobre la gráfica (lo cual permite adaptar los parámetros según el comportamiento que se desee, a lo largo de varias iteraciones).

## Tabla de Errores de Test

+------------------------+----------+----------+----------+
| Momentum\Learning Rate |   0.1    |   0.01   |  0.001   |
+------------------------+----------+----------+----------+
|                      0 | 0.066526 | 0.115962 | 0.123418 |
|                    0.5 | 0.116789 | 0.070222 | 0.118285 |
|                    0.9 | 0.123891 | 0.059511 | 0.116201 |
+------------------------+----------+----------+----------+


# b
El resultado me parece esperable. A medida que se achica el tamaño del conjunto de entrenamiento, se vuelve más fácil para la ANN minimizar el error. A la vez, sobreajusta también con mayor facilidad por lo que dispara el error de test. De nuevo vemos la intuición de "cuantos más datos tengo para entrenar, mejor". Me llama la atención el comportamiento del conjunto de validación, en particular que sea tan grande el error en el de 95% de entrenamiento. Asumo que al ser tan pequeño es muy volátil.

# c
Gracias a Mitchell no tengo que hacer una derivada fea y puedo simplemente multiplicar los pesos por una constante en cada iteración (página 117). Los cambios en el código fueron I/O de gamma, multiplicar los pesos por la constante que dice Mitchell y me tomé el atrevimiento de reemplazar el error de validación (que no existe en este apartado, pues tomamos todos los patrones para entrenar) por la función de error que propone Mitchell (también en la página 117). Entiendo que esto era lo que querías decir con "término de penalización" en el enunciado.

A gamma más grande se dispara el término de penalización (esperable ya que multiplica a la suma del cuadrado de los pesos) y a gammas muy pequeños la penalización no es suficiente para evitar que la red sobreajuste. Me gustó la pinta del gamma = 10^-5, pues es el máximo para el que no se dispara la penalización (y lo razonable es que el error de test sea mayor que el de entrenamiento). Por lo que dije antes, el sobreajuste es más claro en gamma = 10^-8, así que elegí ese como comparación.

# d
(Lo corrí en una notebook viejita y estuvo más de una hora!)

No sabía si usar la mediana (como vengo haciendo en este TP) o la media (como hice en el ejercicio 7 del TP 1), así que hice ambas. No cambió mucho, asumo que por haber hecho 20 iteraciones.

Veo, de forma esperable, que la ANN se adapta de forma similar a ambos problemas. Esto se contrapone al TP 1, donde el dataset diagonal era muy dificil para el espacio de soluciones de c4.5. Este comportamiento denota que la forma de encontrar soluciones de ANN no tiene correlación con diagonal o paralelo.

Por supuesto que en ambos problemas aumentar la dimensión disminuye la densidad de puntos, lo que se refleja en un crecimiento del error de test.

Me llama la atención que le sea más fácil sobreajustar en dimensiones altas para diagonal que para paralelo, no le encuentro explicación.

# e
De nuevo, gracias Mitchell. En las páginas 114 y 115 propone 1-of-n, es decir n neuronas numeradas de salida (donde n es la dimensión). La neurona con mayor output se corresponde con la clase. Además propone no usar 0.1 y 0.9 en vez de 0 y 1 para los patrones ya conocidos, para que sea posible de obtener con neuronas sigmoideas.

Las ventajas son más grados de libertad (más aristas), es decir más poder expresivo de la red y la posibilidad de comprobar cuán segura está la red de un resultado (comprobando la diferencia entre los dos valores de output más altos). Tiene desventajas (aparte de lo no trivial de implementar)? Mitchell no dice ninguna y no se me ocurren.

Me la compliqué bastante para implementar esto:
  1. Pensé que zafaba de editar `bp.c`, así que hice `bp_multiclass_wrapper.py`, que toma los files con sufijo `_original`, convierte la clase en un vector para el input, corre bp y convierte los vectores en clases para el output. Se usa `./bp_multiclass_wrapper fileStem classAmount` (el fileStem no lleva `_original`!)
  2. Me di cuenta que no tenía forma de calcular el error discreto sin tocar bp, así que hice `bp_wrapper.c`, que modifica sólo esa parte.
  3. Para faces, además seguí el consejo de Mitchell de pasar los valores de entrada al intervalo [0,1], para ello usé `faces_normalize.py`, que espera files con sufijo `_not_normalized` y devuelve files con sufijo `_original` (para que los consuma el wrapper).

Estoy positivamente sorprendido por los resultados.

En iris, rápidamente converge y es capaz de clasificar todos los casos de test correctamente. El MSE tarda un poco más en converger pero muestra un comportamiento similar. Asumo que es debido a que hay más casos de train que de test.

En faces, Mitchell promete una precisión del 90% y en mis intentos logré errores de clasificación de entre 8% y 12%. Es claro que Mitchell sabe del tema. Lo que hice debe estar al menos cerca de ser correcto, porque me dio parecido a él.

# f
Agregué el parámetro `BATCHSIZE`. Para hacer minibatch, es necesario agregar un "bucle de batch" dentro del bucle de entrenamiento que recorre los patrones. Dentro del bucle de batch, se acumulan los gradientes. Los pesos de las aristas sólo se modifican al salir del bucle, usando los gradientes acumulados. Luego, se resetean los gradientes a 0.

No esperaba este resultado. Me llama la atención que en training usar batch 1 o 5 sea indistinto. Asumo que el error de 10 es más grande porque el tomar de a varios patrones de entrenamiento dificulta sobreespecializarse. En test no sé qué pasó, esperaba que hubiera un orden claro entre los tres.
