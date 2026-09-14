import csv
import json
import math

def main():
	# Cargar los parámetros entrenados
	with open("./trained_data.json", "r") as f:
		data = json.load(f)
	
	theta0 = data["theta0"]
	theta1 = data["theta1"]

	# Leer datos reales
	kms = []
	prices = []
	with open("./data.csv", "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			kms.append(int(row["km"]))
			prices.append(int(row["price"]))

	m = len(kms)

	# Calcular predicciones y errores
	errors = []
	squared_errors = []
	absolute_errors = []
	
	for i in range(m):
		prediction = theta0 + theta1 * kms[i]
		error = prediction - prices[i]
		errors.append(error)
		squared_errors.append(error ** 2)
		absolute_errors.append(abs(error))

	# MSE: Error Cuadrático Medio
	mse = sum(squared_errors) / m
	
	# RMSE: Raíz del Error Cuadrático Medio
	rmse = math.sqrt(mse)
	
	# MAE: Error Absoluto Medio
	mae = sum(absolute_errors) / m
	
	# R²: Coeficiente de determinación
	mean_price = sum(prices) / m
	ss_tot = sum((p - mean_price) ** 2 for p in prices)
	ss_res = sum(squared_errors)
	r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

	print("=== Métricas de precisión del modelo ===\n")
	print(f"MSE  (Error Cuadrático Medio): {mse:.2f}")
	print(f"RMSE (Raíz del MSE): {rmse:.2f}")
	print(f"MAE  (Error Absoluto Medio): {mae:.2f}")
	print(f"R²   (Coeficiente de determinación): {r2:.4f}")
	print(f"\nInterpretación:")
	print(f"- El modelo se equivoca en promedio {mae:.0f}€ por predicción")
	print(f"- R² = {r2:.2%} de la varianza del precio es explicada por el modelo")

if __name__ == "__main__":
	main()
