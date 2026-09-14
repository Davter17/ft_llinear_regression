import csv
import json
import matplotlib.pyplot as plt

def main():
	# Cargar los parámetros entrenados desde el archivo JSON
	with open("./trained_data.json", "r") as f:
		data = json.load(f)
	
	theta0 = data["theta0"]
	theta1 = data["theta1"]

	# Solicitar kilometraje al usuario
	try:
		km = int(input("Introduce el kilometraje del coche: "))
	except ValueError:
		km = 0

	# Calcular precio usando la hipótesis: price = theta0 + theta1 * km
	price = theta0 + theta1 * km
	print(f"El precio estimado es: {price:.2f}")

	# Leer datos originales para visualización
	kms = []
	prices = []
	with open("./data.csv", "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			kms.append(int(row["km"]))
			prices.append(int(row["price"]))

	# Calcular predicciones para todos los puntos de datos
	predictions = [theta0 + theta1 * km for km in kms]

	# Ordenar puntos para dibujar la línea de regresión correctamente
	combined = sorted(zip(kms, predictions))
	km_sorted, pred_sorted = zip(*combined)

	# Visualizar resultados: datos reales, predicción del usuario y línea de regresión
	plt.scatter(kms, prices, color='blue', label='Datos reales')
	plt.scatter(km, price, color='red', label='Predicción', s=100)
	plt.plot(km_sorted, pred_sorted, color='green', label='Línea de regresión')
	plt.xlabel('Kilometraje')
	plt.ylabel('Precio')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()
