# ft_linear_regression

Implementation of a simple linear regression algorithm from scratch using gradient descent to predict car prices based on mileage.

## Overview

This project predicts car prices using the linear hypothesis:

```
price = θ₀ + θ₁ × mileage
```

Where:
- `θ₀` (theta0) = intercept
- `θ₁` (theta1) = slope

The model is trained using gradient descent with min-max normalization for better convergence.

## Project Structure

```
├── data.csv           # Dataset with 24 cars (mileage and price)
├── training.py        # Train the model and save parameters
├── predict.py         # Predict price and visualize results
├── score.py           # Calculate model accuracy metrics
└── trained_data.json  # Trained parameters (generated)
```

## Installation

```bash
pip install matplotlib
```

## Usage

### 1. Train the model

```bash
python training.py
```

Output:
```
theta0: 8481.17
theta1: -0.0213
MSE: 445727.42
```

### 2. Make predictions

```bash
python predict.py
```

Enter the mileage when prompted:
```
Introduce el kilometraje del coche: 100000
El precio estimado es: 6353.80
```

A graph will display showing:
- Blue dots: real data
- Green line: regression line
- Red dot: your prediction

### 3. Evaluate model accuracy

```bash
python score.py
```

Output:
```
=== Métricas de precisión del modelo ===

MSE  (Error Cuadrático Medio): 445727.42
RMSE (Raíz del MSE): 667.59
MAE  (Error Absoluto Medio): 523.41
R²   (Coeficiente de determinación): 0.7823

Interpretación:
- El modelo se equivoca en promedio 523€ por predicción
- R² = 78.23% de la varianza del precio es explicada por el modelo
```

## Algorithm

### Gradient Descent

The algorithm iteratively adjusts θ₀ and θ₁ to minimize prediction error:

```
θ₀ = θ₀ - α × (1/m) × Σ(ŷ - y)
θ₁ = θ₁ - α × (1/m) × Σ(ŷ - y) × x
```

Where:
- `α` = learning rate (0.1)
- `m` = number of samples
- `ŷ` = predicted value
- `y` = actual value

### Normalization

Data is normalized using min-max scaling for better convergence:

```
x_norm = (x - x_min) / (x_max - x_min)
```

After training, parameters are denormalized to work with original values.

## Metrics

The `score.py` program calculates:

- **MSE** (Mean Squared Error): Average squared difference between predictions and actual values
- **RMSE** (Root Mean Squared Error): Square root of MSE, in the same units as the target
- **MAE** (Mean Absolute Error): Average absolute difference
- **R²** (Coefficient of Determination): Proportion of variance explained by the model (0-1)

## Dataset

The `data.csv` file contains 24 cars with:
- **km**: mileage (22,899 - 240,000 km)
- **price**: price in euros (3,650 - 8,290 €)

## Configuration

In `training.py`, you can adjust:
- **lr** (learning rate): 0.1
- **iterations**: 1000

A learning rate too high may cause divergence, too low will converge slowly.

## License

This project is part of the 42 school curriculum (OuterCore).
