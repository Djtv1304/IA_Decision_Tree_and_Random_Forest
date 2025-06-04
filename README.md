# Informe Comparativo: Árbol de Decisión vs Random Forest - Predicción de Rotación de Empleados

## 1. Introducción y Objetivo

El presente informe tiene como finalidad describir de manera detallada y estructurada el desarrollo, entrenamiento y evaluación de dos modelos de aprendizaje automático: **Árbol de Decisión** y **Random Forest**, aplicados a datos de recursos humanos para predecir la **rotación de empleados** (`Attrition`). El objetivo concreto es:

* Comparar el rendimiento de ambos algoritmos utilizando variables demográficas y económicas (edad del empleado e ingreso mensual) para clasificar correctamente si un empleado permanecerá en la empresa (0) o la abandonará (1).
* Documentar el procedimiento completo de preprocesamiento, entrenamiento, evaluación y visualización de ambos modelos, resaltando las métricas clave, especialmente la **precisión (accuracy)**.
* Establecer una comparación directa entre ambos enfoques para determinar cuál ofrece mejor rendimiento predictivo.

## 2. Descripción del Conjunto de Datos

El dataset emplea dos variables predictoras y una variable objetivo, con los siguientes atributos:

| Variable           | Descripción                                                                      | Tipo                 |
| ------------------ | -------------------------------------------------------------------------------- | -------------------- |
| **Age**            | Edad del empleado (en años).                                                     | Numérica             |
| **MonthlyIncome**  | Ingreso mensual del empleado (en unidades monetarias).                          | Numérica             |
| **Attrition**      | Indicador binario: 1 si el empleado abandona la empresa, 0 si permanece.        | Categórica (Binaria) |

### Ejemplos de Datos

| Age | MonthlyIncome | Attrition |
| --- | ------------- | --------- |
| 41  | 5993          | 1         |
| 49  | 5130          | 0         |
| 37  | 2090          | 1         |
| 33  | 2909          | 0         |
| 27  | 3468          | 0         |

## 3. Preprocesamiento de Datos

1. **Carga del CSV**
   ```python
   dataset = pd.read_csv('Employee_Attrition_Modified.csv')
   ```
   El archivo contiene múltiples registros de empleados con las columnas `Age`, `MonthlyIncome` y `Attrition`.

2. **Separación de características (X) y variable objetivo (y)**
   ```python
   X = dataset.iloc[:, :-1].values  # [Age, MonthlyIncome]
   y = dataset.iloc[:, -1].values   # [Attrition]
   ```

3. **División en Conjunto de Entrenamiento y Prueba**
   * **Proporción**: 75% entrenamiento / 25% prueba.
   * **Aleatoriedad fija** (`random_state = 0`) para reproducibilidad.
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=0
   )
   ```

4. **Escalado de Características**
   * Se utiliza **StandardScaler** para normalizar `Age` y `MonthlyIncome`.
   * Aunque los árboles de decisión no requieren escalado, se mantiene para consistencia con otros modelos.
   ```python
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test  = sc.transform(X_test)
   ```

## 4. Entrenamiento de los Modelos

### 4.1 Árbol de Decisión

1. **Configuración del Clasificador**
   ```python
   classifier = DecisionTreeClassifier(
       criterion='entropy', 
       random_state=0
   )
   classifier.fit(X_train, y_train)
   ```
   * **Criterio de División**: `entropy` - mide el nivel de desorden para encontrar las mejores divisiones.
   * **Principio**: Construye un árbol donde cada nodo representa una decisión basada en una característica, dividiendo los datos hasta alcanzar hojas puras o criterios de parada.

### 4.2 Random Forest

1. **Configuración del Clasificador**
   ```python
   classifier = RandomForestClassifier(
       n_estimators=10, 
       criterion='entropy', 
       random_state=4500
   )
   classifier.fit(X_train, y_train)
   ```
   * **Número de Estimadores**: 10 árboles de decisión independientes.
   * **Criterio de División**: `entropy` para consistencia con el árbol individual.
   * **Principio**: Combina las predicciones de múltiples árboles mediante votación mayoritaria, reduciendo overfitting y mejorando generalización.

2. **Predicción de Ejemplo Puntual**
   * Se proyecta un caso de prueba: **edad = 30 años**, **ingreso mensual = 87,000**.
   * Ambos modelos devuelven `0` o `1` indicando si se clasifica como permanencia o rotación.

## 5. Evaluación y Métricas Comparativas

### 5.1 Resultados del Árbol de Decisión

#### Matriz de Confusión - Árbol de Decisión
|                              | Predicción = 0 (No Rotación) | Predicción = 1 (Rotación) |
| ---------------------------- | ---------------------------: | ------------------------: |
| **Real = 0 (No Rotación)**   |                      **174** |                    **35** |
| **Real = 1 (Rotación)**      |                       **33** |                     **9** |

* **Verdaderos Negativos (TN)** = 174
* **Falsos Positivos (FP)** = 35  
* **Falsos Negativos (FN)** = 33
* **Verdaderos Positivos (TP)** = 9

#### Métrica de Precisión - Árbol de Decisión
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{9 + 174}{9 + 174 + 35 + 33} = \frac{183}{251} = 0{,}7291
$$

* **Valor**: 0.7291 ⇒ **72,91%**

### 5.2 Resultados del Random Forest

#### Matriz de Confusión - Random Forest
|                              | Predicción = 0 (No Rotación) | Predicción = 1 (Rotación) |
| ---------------------------- | ---------------------------: | ------------------------: |
| **Real = 0 (No Rotación)**   |                      **191** |                    **18** |
| **Real = 1 (Rotación)**      |                       **36** |                     **6** |

* **Verdaderos Negativos (TN)** = 191
* **Falsos Positivos (FP)** = 18  
* **Falsos Negativos (FN)** = 36
* **Verdaderos Positivos (TP)** = 6

#### Métrica de Precisión - Random Forest
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{6 + 191}{6 + 191 + 18 + 36} = \frac{197}{251} = 0{,}7849
$$

* **Valor**: 0.7849 ⇒ **78,49%**

### 5.3 Comparación Directa de Rendimiento

| Métrica                    | Árbol de Decisión | Random Forest | Diferencia  | Ganador      |
| -------------------------- | ----------------: | ------------: | ----------: | ------------ |
| **Accuracy**               |            72,91% |        78,49% |      +5,58% | Random Forest |
| **Verdaderos Positivos**   |                 9 |             6 |          -3 | Árbol Decisión |
| **Verdaderos Negativos**   |               174 |           191 |         +17 | Random Forest |
| **Falsos Positivos**       |                35 |            18 |         -17 | Random Forest |
| **Falsos Negativos**       |                33 |            36 |          +3 | Árbol Decisión |

**Análisis Clave**:
* **Random Forest supera al Árbol de Decisión en accuracy general** por 5,58 puntos porcentuales.
* **Random Forest es significativamente mejor identificando empleados que NO rotarán** (191 vs 174 verdaderos negativos).
* **Random Forest comete menos errores de falsos positivos** (18 vs 35), reduciendo costos de programas de retención innecesarios.
* **Árbol de Decisión es ligeramente mejor detectando rotación real** (9 vs 6 verdaderos positivos), pero la diferencia es marginal.

## 6. Visualización Comparativa de los Resultados

### 6.1 Árbol de Decisión (Conjunto de Prueba)

![image](https://github.com/user-attachments/assets/2e4aa4ee-6063-438a-8e4a-271511b458e2)

**Descripción**:
* El plano se graficó usando las características originales (`Edad` ↔ `Age`, `Ingreso Mensual` ↔ `MonthlyIncome`).
* La región en **verde** corresponde a la clase `1` (rotación), y la región en **rojo** a la clase `0` (permanencia).
* Los puntos rojos representan observaciones de empleados que permanecen, y los verdes los que abandonan la empresa.

**Observaciones del Árbol de Decisión**:
* Las **fronteras de decisión son rectangulares y rígidas**, característica típica de los árboles de decisión que realizan divisiones orthogonales.
* Se observan **múltiples regiones fragmentadas**, especialmente en la zona de ingresos medios (5,000-12,000).
* La **complejidad visual** sugiere que el modelo puede estar capturando ruido específico del conjunto de entrenamiento.
* **Overfitting potencial**: Las divisiones muy específicas pueden no generalizar bien a nuevos datos.

### 6.2 Random Forest (Conjunto de Prueba)

![image](https://github.com/user-attachments/assets/289396c1-a61d-4053-93a9-2487ef60d7cc)

**Descripción**:
* Misma lógica gráfica que en el árbol de decisión individual.
* Las fronteras muestran el resultado de la **votación promedio** de los 10 árboles del ensemble.

**Observaciones del Random Forest**:
* Las **fronteras de decisión son más suaves y regulares** comparadas con el árbol individual.
* **Menor fragmentación** en las regiones de decisión, indicando mayor estabilidad.
* La **región roja (no rotación) es más cohesiva**, especialmente en la zona de ingresos altos.
* **Mejor balance** entre complejidad y generalización, evidenciando el efecto de regularización del ensemble.

### 6.3 Comparación Visual Directa

| Aspecto                    | Árbol de Decisión         | Random Forest             |
| -------------------------- | ------------------------- | ------------------------- |
| **Fronteras**              | Rectangulares, rígidas    | Suaves, regulares         |
| **Fragmentación**          | Alta, múltiples regiones  | Baja, regiones cohesivas  |
| **Complejidad Visual**     | Muy alta                  | Moderada                  |
| **Indicios de Overfitting**| Evidentes                 | Reducidos                 |
| **Estabilidad**            | Baja                      | Alta                      |

## 7. Análisis Detallado del Rendimiento

### 7.1 Métricas Adicionales por Modelo

#### Árbol de Decisión - Análisis Detallado
1. **Tasa de Verdaderos Positivos (Sensibilidad)**
   $$
   \text{Sensibilidad} = \frac{TP}{TP + FN} = \frac{9}{9 + 33} = 0{,}2143 \quad(21{,}43\%)
   $$

2. **Tasa de Verdaderos Negativos (Especificidad)**
   $$
   \text{Especificidad} = \frac{TN}{TN + FP} = \frac{174}{174 + 35} = 0{,}8326 \quad(83{,}26\%)
   $$

#### Random Forest - Análisis Detallado
1. **Tasa de Verdaderos Positivos (Sensibilidad)**
   $$
   \text{Sensibilidad} = \frac{TP}{TP + FN} = \frac{6}{6 + 36} = 0{,}1429 \quad(14{,}29\%)
   $$

2. **Tasa de Verdaderos Negativos (Especificidad)**
   $$
   \text{Especificidad} = \frac{TN}{TN + FP} = \frac{191}{191 + 18} = 0{,}9139 \quad(91{,}39\%)
   $$

### 7.2 Comparación de Sensibilidad y Especificidad

| Métrica           | Árbol de Decisión | Random Forest | Diferencia  | Ganador         |
| ----------------- | ----------------: | ------------: | ----------: | --------------- |
| **Sensibilidad**  |            21,43% |        14,29% |      -7,14% | Árbol Decisión  |
| **Especificidad** |            83,26% |        91,39% |      +8,13% | Random Forest   |

**Interpretación Crítica**:
* **Random Forest sacrifica sensibilidad por especificidad**: Detecta menos empleados que realmente se van, pero es mucho mejor identificando los que se quedan.
* **Árbol de Decisión es más equilibrado** pero con rendimiento general inferior.
* **Ambos modelos sufren del problema de desbalance de clases**, pero Random Forest maneja mejor la clase mayoritaria.

### 7.3 Impacto Empresarial de las Diferencias

#### Costos de Falsos Negativos (No detectar rotación real)
* **Árbol de Decisión**: 33 empleados en riesgo no detectados
* **Random Forest**: 36 empleados en riesgo no detectados
* **Diferencia**: Random Forest pierde 3 detecciones adicionales

#### Costos de Falsos Positivos (Alarmas falsas)
* **Árbol de Decisión**: 35 empleados mal clasificados como en riesgo
* **Random Forest**: 18 empleados mal clasificados como en riesgo
* **Diferencia**: Random Forest reduce en 17 las alarmas falsas

**Balance Costo-Beneficio**:
* **Random Forest es superior** considerando que los costos de programas de retención innecesarios (falsos positivos) suelen ser menores que los costos de rotación no detectada.
* La **reducción de 17 falsos positivos** compensa la pérdida de 3 detecciones verdaderas.

## 8. Ventajas y Desventajas Comparativas

### 8.1 Árbol de Decisión

#### Ventajas
* **Interpretabilidad Superior**: El modelo es completamente explicable y auditable.
* **Rapidez de Entrenamiento**: Entrenamiento más rápido con un solo árbol.
* **Mejor Sensibilidad**: Detecta ligeramente más casos de rotación real.
* **Sin Hiperparámetros Complejos**: Menos parámetros que ajustar.

#### Desventajas
* **Alto Overfitting**: Memoriza patrones específicos del entrenamiento.
* **Inestabilidad**: Cambios pequeños en datos pueden alterar drasticamente el árbol.
* **Fronteras Rígidas**: Divisiones rectangulares pueden ser demasiado simplistas.
* **Mayor Tasa de Falsos Positivos**: Genera más alarmas falsas costosas.

### 8.2 Random Forest

#### Ventajas
* **Mayor Accuracy General**: 5,58% superior al árbol individual.
* **Mejor Generalización**: Reduce overfitting mediante averaging de múltiples árboles.
* **Mayor Estabilidad**: Menos sensible a variaciones en los datos.
* **Mejor Especificidad**: Excelente identificando empleados que permanecerán.
* **Menor Tasa de Falsos Positivos**: Reduce costos de programas de retención innecesarios.

#### Desventajas
* **Menor Interpretabilidad**: Más difícil explicar decisiones individuales.
* **Mayor Costo Computacional**: Requiere entrenar y evaluar múltiples árboles.
* **Menor Sensibilidad**: Detecta menos casos de rotación real.
* **Más Hiperparámetros**: Número de árboles, profundidad máxima, etc.

## 9. Recomendaciones Específicas por Modelo

### 9.1 Para Mejorar el Árbol de Decisión
1. **Control de Overfitting**
   * **Poda (Pruning)**: Limitar la profundidad máxima (`max_depth`).
   * **Mínimo de muestras**: Establecer `min_samples_split` y `min_samples_leaf`.
   * **Validación Cruzada**: Para seleccionar hiperparámetros óptimos.

2. **Balanceo de Clases**
   * **Class Weight**: Usar `class_weight='balanced'` para penalizar más los errores en la clase minoritaria.
   * **Threshold Tuning**: Ajustar el umbral de decisión para mejorar sensibilidad.

### 9.2 Para Optimizar Random Forest
1. **Aumento de Estimadores**
   * **Más Árboles**: Incrementar `n_estimators` de 10 a 50-100 para mayor estabilidad.
   * **Análisis de Convergencia**: Determinar el número óptimo donde la mejora se estabiliza.

2. **Mejora de Sensibilidad**
   * **Class Weight Balanceado**: Aplicar `class_weight='balanced'`.
   * **Threshold Optimization**: Usar curvas ROC para encontrar el umbral óptimo.
   * **Sampling Techniques**: SMOTE para generar ejemplos sintéticos de rotación.

### 9.3 Recomendación General
**Para el contexto empresarial de predicción de rotación, se recomienda Random Forest** por las siguientes razones:

1. **Mayor Accuracy Global** (78,49% vs 72,91%)
2. **Mejor Manejo de la Clase Mayoritaria** (91,39% especificidad)
3. **Menor Tasa de Falsos Positivos** (reducción de costos operativos)
4. **Mayor Estabilidad y Confiabilidad** del modelo en producción

Sin embargo, **se debe trabajar en mejorar la sensibilidad** mediante técnicas de balanceeo de clases y optimización de umbrales.

## 10. Conclusiones

* **Random Forest supera al Árbol de Decisión** con una precisión superior del **78,49% vs 72,91%**, representando una mejora de **5,58 puntos porcentuales**.

* La **matriz de confusión comparativa** revela que Random Forest es significativamente mejor manejando la **clase mayoritaria** (no rotación), con **91,39% de especificidad vs 83,26%** del árbol individual.

* **Ambos modelos sufren de baja sensibilidad** para detectar rotación real (Random Forest: 14,29%, Árbol de Decisión: 21,43%), evidenciando el **impacto del desbalance de clases** en el dataset.

* Las **visualizaciones** demuestran que Random Forest genera **fronteras de decisión más suaves y estables**, mientras que el Árbol de Decisión muestra **fragmentación excesiva** indicativa de overfitting.

* Desde una **perspectiva de costo-beneficio empresarial**, Random Forest es superior al reducir falsos positivos (17 menos) a costa de solo 3 falsos negativos adicionales.

* Para **implementación en producción**, se recomienda **Random Forest con optimizaciones adicionales**: incrementar el número de estimadores, aplicar balanceeo de clases, y ajustar umbrales de decisión para mejorar la detección de rotación real.

* **Futuras mejoras** deberían enfocarse en técnicas de manejo de datos desbalanceados, incorporación de variables adicionales, y exploración de modelos más avanzados como Gradient Boosting para maximizar tanto la precisión general como la sensibilidad específica para detección de rotación.

**Autor:** Diego Toscano  
**Contacto:** [diego.toscano@udla.edu.ec](mailto:diego.toscano@udla.edu.ec)
