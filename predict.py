import csv
import json
import matplotlib.pyplot as plt

def main():
	# Cargar los parámetros entrenados desde el archivo JSON
	try:
		with open("./trained_data.json", "r") as f:
			data = json.load(f)
		theta0 = data["theta0"]
		theta1 = data["theta1"]
	except FileNotFoundError:
		print("Warning: trained_data.json not found. Using default values (theta0=0, theta1=0).")
		print("Run training.py first to get accurate predictions.")
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

	# Solicitar kilometraje al usuario
	while True:
		try:
			km_input = input("Enter the car mileage: ")
			km = int(km_input)
			if km < 0:
				print("Error: mileage cannot be negative")
				continue
			break
		except ValueError:
			print("Error: please enter a valid number")
			continue

	# Calcular precio usando la hipótesis: price = theta0 + theta1 * km
	price = theta0 + theta1 * km
	print(f"The estimated price is: {price:.2f}")

	# Leer datos originales para visualización
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

	# Calcular predicciones para todos los puntos de datos
	predictions = [theta0 + theta1 * km for km in kms]

	# Ordenar puntos para dibujar la línea de regresión correctamente
	combined = sorted(zip(kms, predictions))
	km_sorted, pred_sorted = zip(*combined)

	# Visualizar resultados: datos reales, predicción del usuario y línea de regresión
	plt.scatter(kms, prices, color='blue', label='Real data')
	plt.scatter(km, price, color='red', label='Prediction', s=100)
	plt.plot(km_sorted, pred_sorted, color='green', label='Regression line')
	plt.xlabel('Mileage')
	plt.ylabel('Price')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()
