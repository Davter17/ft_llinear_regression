import csv
import json
import math

def main():
	# Cargar los parámetros entrenados
	try:
		with open("./trained_data.json", "r") as f:
			data = json.load(f)
		theta0 = data["theta0"]
		theta1 = data["theta1"]
	except FileNotFoundError:
		print("Warning: trained_data.json not found. Using default values (theta0=0, theta1=0).")
		print("Run training.py first to get accurate metrics.")
		theta0 = 0.0
		theta1 = 0.0
	except PermissionError:
		print("Error: you don't have permission to read trained_data.json")
		return
	except json.JSONDecodeError as e:
		print(f"Error: trained_data.json is corrupted - {e}")
		return
	except KeyError as e:
		print(f"Error: trained_data.json is missing required field - {e}")
		return

	# Leer datos reales
	kms = []
	prices = []
	try:
		with open("./data.csv", "r") as f:
			reader = csv.DictReader(f)
			for i, row in enumerate(reader, 1):
				try:
					km = int(row["km"])
					price = int(row["price"])
					if km < 0 or price < 0:
						print(f"Error: row {i} has negative values (km={km}, price={price})")
						return
					kms.append(km)
					prices.append(price)
				except (ValueError, KeyError) as e:
					print(f"Error: row {i} has invalid data - {e}")
					return
	except FileNotFoundError:
		print("Error: data.csv not found")
		return
	except PermissionError:
		print("Error: you don't have permission to read data.csv")
		return

	lenDatas = len(kms)

	# Calcular predicciones y errores
	errors = []
	squared_errors = []
	absolute_errors = []
	
	for i in range(lenDatas):
		prediction = theta0 + theta1 * kms[i]
		error = prediction - prices[i]
		errors.append(error)
		squared_errors.append(error ** 2)
		absolute_errors.append(abs(error))

	# MSE: Error Cuadrático Medio
	mse = sum(squared_errors) / lenDatas
	
	# RMSE: Raíz del Error Cuadrático Medio
	rmse = math.sqrt(mse)
	
	# MAE: Error Absoluto Medio
	mae = sum(absolute_errors) / lenDatas
	
	# R²: Coeficiente de determinación
	mean_price = sum(prices) / lenDatas
	ss_tot = sum((p - mean_price) ** 2 for p in prices)
	ss_res = sum(squared_errors)
	r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

	print("=== Model accuracy metrics ===\n")
	print(f"MSE  (Mean Squared Error): {mse:.2f}")
	print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
	print(f"MAE  (Mean Absolute Error): {mae:.2f}")
	print(f"R²   (Coefficient of Determination): {r2:.4f}")
	print(f"\nInterpretation:")
	print(f"- The model is off by an average of {mae:.0f}€ per prediction")
	print(f"- R² = {r2:.2%} of the price variance is explained by the model")

if __name__ == "__main__":
	main()
