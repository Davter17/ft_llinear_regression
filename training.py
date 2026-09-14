import csv
import json

def main():
	# Leer datos del archivo CSV
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
	
	# Normalizar datos usando min-max para mejorar convergencia del gradiente
	km_min = min(kms)
	km_max = max(kms)
	price_min = min(prices)
	price_max = max(prices)
	
	kms_norm = [(k_act - km_min) / (km_max - km_min) for k_act in kms]
	prices_norm = [(p_act - price_min) / (price_max - price_min) for p_act in prices]

	# Inicializar parámetros del modelo
	theta0 = 0.0
	theta1 = 0.0
	lr = 0.1
	iterations = 1000

	# Gradiente descendente: ajustar theta0 y theta1 iterativamente
	for i in range(iterations):
		sumError0 = 0.0
		sumError1 = 0.0
		for j in range(lenDatas):
			estimate = theta0 + theta1 * kms_norm[j]
			errorPrice = estimate - prices_norm[j]
			sumError0 += errorPrice
			sumError1 += errorPrice * kms_norm[j]
		theta0 -= lr * (1 / lenDatas) * sumError0
		theta1 -= lr * (1 / lenDatas) * sumError1

	# Desnormalizar los thetas para usarlos con valores originales
	theta0_orig = theta0 * (price_max - price_min) + price_min - theta1 * (km_min / (km_max - km_min)) * (price_max - price_min)
	theta1_orig = theta1 * (price_max - price_min) / (km_max - km_min)

	# Calcular error cuadrático medio (MSE)
	mse = 0.0
	for j in range(lenDatas):
		estimate = theta0_orig + theta1_orig * kms[j]
		mse += (estimate - prices[j]) ** 2
	mse /= lenDatas

	print(f"theta0: {theta0_orig}")
	print(f"theta1: {theta1_orig}")
	print(f"MSE: {mse}")

	# Guardar los parámetros entrenados en formato JSON
	data = {"theta0": theta0_orig, "theta1": theta1_orig}
	try:
		with open("./trained_data.json", "w") as f:
			json.dump(data, f)
	except PermissionError:
		print("Error: you don't have permission to write trained_data.json")
		return

if __name__ == "__main__":
	main()
