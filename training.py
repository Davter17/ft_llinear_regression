import csv
import json

def main():
	# Leer datos del archivo CSV
	kms = []
	prices = []
	with open("./data.csv", "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			kms.append(int(row["km"]))
			prices.append(int(row["price"]))

	m = len(kms)
	
	# Normalizar datos usando min-max para mejorar convergencia del gradiente
	km_min = min(kms)
	km_max = max(kms)
	price_min = min(prices)
	price_max = max(prices)
	
	kms_norm = [(k - km_min) / (km_max - km_min) for k in kms]
	prices_norm = [(p - price_min) / (price_max - price_min) for p in prices]

	# Inicializar parámetros del modelo
	theta0 = 0.0
	theta1 = 0.0
	lr = 0.1
	iterations = 1000

	# Gradiente descendente: ajustar theta0 y theta1 iterativamente
	for i in range(iterations):
		sum0 = 0.0
		sum1 = 0.0
		for j in range(m):
			estimate = theta0 + theta1 * kms_norm[j]
			error = estimate - prices_norm[j]
			sum0 += error
			sum1 += error * kms_norm[j]
		tmp0 = lr * (1 / m) * sum0
		tmp1 = lr * (1 / m) * sum1
		theta0 -= tmp0
		theta1 -= tmp1

	# Desnormalizar los thetas para usarlos con valores originales
	theta0_orig = theta0 * (price_max - price_min) + price_min - theta1 * (km_min / (km_max - km_min)) * (price_max - price_min)
	theta1_orig = theta1 * (price_max - price_min) / (km_max - km_min)

	# Calcular error cuadrático medio (MSE)
	mse = 0.0
	for j in range(m):
		estimate = theta0_orig + theta1_orig * kms[j]
		mse += (estimate - prices[j]) ** 2
	mse /= m

	print(f"theta0: {theta0_orig}")
	print(f"theta1: {theta1_orig}")
	print(f"MSE: {mse}")

	# Guardar los parámetros entrenados en formato JSON
	data = {"theta0": theta0_orig, "theta1": theta1_orig}
	with open("./trained_data.json", "w") as f:
		json.dump(data, f)

if __name__ == "__main__":
	main()
