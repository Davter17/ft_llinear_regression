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

### Option 1: Using a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib
```

### Option 2: System package

```bash
sudo apt install python3-matplotlib
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
Enter the car mileage: 100000
The estimated price is: 6353.80
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
=== Model accuracy metrics ===

MSE  (Mean Squared Error): 445727.42
RMSE (Root Mean Squared Error): 667.59
MAE  (Mean Absolute Error): 523.41
R²   (Coefficient of Determination): 0.7823

Interpretation:
- The model is off by an average of 523€ per prediction
- R² = 78.23% of the price variance is explained by the model
```

## Error Handling

The programs include comprehensive error handling:

- **File errors**: Clear messages if `data.csv` or `trained_data.json` are missing or inaccessible
- **Data validation**: Invalid data (non-numeric, negative values) in CSV files is detected and reported with row numbers
- **Input validation**: Negative or non-numeric mileage inputs are rejected with helpful error messages
- **JSON validation**: Corrupted or malformed `trained_data.json` files are detected

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

The `score.py` program calculates four key metrics to evaluate model performance:

### MSE (Mean Squared Error)
Average of squared differences between predictions and actual values. By squaring errors, it penalizes large errors more heavily than small ones. Useful for detecting outliers, but hard to interpret because it's in squared units (€²). **Lower values = better model.**

### RMSE (Root Mean Squared Error)
Square root of MSE. It's in the same units as the target variable (€), making it more interpretable than MSE. Tells you how much predictions deviate on average. **Lower values = better model.**

### MAE (Mean Absolute Error)
Average of absolute errors (without squaring). Tells you how much the model is wrong per prediction, in euros. More robust to outliers than MSE/RMSE. **Lower values = better model.**

### R² (Coefficient of Determination)
Proportion of price variance explained by the model. Ranges from 0 to 1 (can be negative if model performs worse than predicting the mean). R²=1 means perfect prediction, R²=0 means the model explains nothing. For example, R²=0.78 means 78% of price variation is explained by mileage. **Higher values = better model.**

### Interpretation Example
If MAE=523 and R²=0.78:
- The model is off by an average of 523€ per prediction
- 78% of price variance is explained by mileage
- The remaining 22% is due to other factors (age, condition, brand, etc.)

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
