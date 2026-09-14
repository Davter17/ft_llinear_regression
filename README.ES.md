# ft_linear_regression

Implementación de un algoritmo de regresión lineal simple desde cero usando gradiente descendente para predecir precios de coches según su kilometraje.

## Descripción general

Este proyecto predice precios de coches usando la hipótesis lineal:

```
precio = θ₀ + θ₁ × kilometraje
```

Donde:
- `θ₀` (theta0) = intersección
- `θ₁` (theta1) = pendiente

El modelo se entrena usando gradiente descendente con normalización min-max para una mejor convergencia.

## Estructura del proyecto

```
├── data.csv           # Dataset con 24 coches (kilometraje y precio)
├── training.py        # Entrena el modelo y guarda los parámetros
├── predict.py         # Predice el precio y visualiza los resultados
├── score.py           # Calcula las métricas de precisión del modelo
└── trained_data.json  # Parámetros entrenados (generado automáticamente)
```

## Instalación

### Opción 1: Usando un entorno virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib
```

### Opción 2: Paquete del sistema

```bash
sudo apt install python3-matplotlib
```

## Uso

### 1. Entrenar el modelo

```bash
python training.py
```

Salida:
```
theta0: 8481.17
theta1: -0.0213
MSE: 445727.42
```

### 2. Hacer predicciones

```bash
python predict.py
```

Introduce el kilometraje cuando se te pida:
```
Enter the car mileage: 100000
The estimated price is: 6353.80
```

Se mostrará una gráfica con:
- Puntos azules: datos reales
- Línea verde: línea de regresión
- Punto rojo: tu predicción

### 3. Evaluar la precisión del modelo

```bash
python score.py
```

Salida:
```
=== Model accuracy metrics ===

MSE  (Mean Squared Error): 445727.42
RMSE (Root Mean Squared Error): 667.59
MAE  (Mean Absolute Error): 523.41
R²   (Coefficient of Determination): 0.7823

Interpretation:
- The model is off by an average of 523€ per prediction
- R² = 78.23% of the price variance is explained by the model
```

## Manejo de errores

Los programas incluyen manejo de errores completo:

- **Errores de archivo**: Mensajes claros si `data.csv` o `trained_data.json` faltan o no son accesibles
- **Validación de datos**: Se detectan y reportan datos inválidos (no numéricos, valores negativos) en archivos CSV con números de fila
- **Validación de entrada**: Se rechazan entradas de kilometraje negativas o no numéricas con mensajes de error útiles
- **Validación de JSON**: Se detectan archivos `trained_data.json` corruptos o malformados

## Algoritmo

### Gradiente descendente

El algoritmo ajusta iterativamente θ₀ y θ₁ para minimizar el error de predicción:

```
θ₀ = θ₀ - α × (1/m) × Σ(ŷ - y)
θ₁ = θ₁ - α × (1/m) × Σ(ŷ - y) × x
```

Donde:
- `α` = tasa de aprendizaje (0.1)
- `m` = número de muestras
- `ŷ` = valor predicho
- `y` = valor real

### Normalización

Los datos se normalizan usando escalado min-max para una mejor convergencia:

```
x_norm = (x - x_min) / (x_max - x_min)
```

Después del entrenamiento, los parámetros se desnormalizan para trabajar con valores originales.

## Métricas

El programa `score.py` calcula cuatro métricas clave para evaluar el rendimiento del modelo:

### MSE (Error Cuadrático Medio)
Promedio de las diferencias al cuadrado entre predicciones y valores reales. Al elevar al cuadrado los errores, penaliza más los errores grandes que los pequeños. Útil para detectar outliers, pero difícil de interpretar porque está en unidades cuadradas (€²). **Valores más bajos = mejor modelo.**

### RMSE (Raíz del Error Cuadrático Medio)
Raíz cuadrada del MSE. Está en las mismas unidades que la variable objetivo (€), por lo que es más interpretable que el MSE. Te dice cuánto se desvían las predicciones en promedio. **Valores más bajos = mejor modelo.**

### MAE (Error Absoluto Medio)
Promedio de los errores absolutos (sin elevar al cuadrado). Te dice cuánto se equivoca el modelo en promedio por predicción, en euros. Es más robusto a outliers que MSE/RMSE. **Valores más bajos = mejor modelo.**

### R² (Coeficiente de Determinación)
Proporción de la varianza del precio que explica el modelo. Va de 0 a 1 (puede ser negativo si el modelo funciona peor que predecir la media). R²=1 significa predicción perfecta, R²=0 significa que el modelo no explica nada. Por ejemplo, R²=0.78 significa que el 78% de la variación del precio se explica por el kilometraje. **Valores más altos = mejor modelo.**

### Ejemplo de interpretación
Si MAE=523 y R²=0.78:
- El modelo se equivoca en promedio 523€ por predicción
- El 78% de la variación del precio se explica por el kilometraje
- El 22% restante se debe a otros factores (edad, estado, marca, etc.)

## Dataset

El archivo `data.csv` contiene 24 coches con:
- **km**: kilometraje (22.899 - 240.000 km)
- **price**: precio en euros (3.650 - 8.290 €)

## Configuración

En `training.py` puedes ajustar:
- **lr** (tasa de aprendizaje): 0.1
- **iterations**: 1000

Una tasa de aprendizaje muy alta puede causar divergencia, una muy baja hará que converja lentamente.

## Licencia

Este proyecto forma parte del plan de estudios de 42 (OuterCore).
