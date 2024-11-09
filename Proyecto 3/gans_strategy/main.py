import pandas as pd
from utils.utils import load_data, create_gan_model, train_gan, generate_synthetic_data

# Paso 1: Cargar datos históricos
data = load_data("data/historical_data.csv")  # Cambia el nombre del archivo según el que tengas

# Paso 2: Crear y entrenar el modelo GAN
gan_model = create_gan_model(input_shape=data.shape[1])  # Define el input_shape según tus datos
train_gan(gan_model, data, epochs=1000)  # Ajusta los epochs según sea necesario

# Paso 3: Generar datos sintéticos
synthetic_data = generate_synthetic_data(gan_model, num_scenarios=100)
synthetic_data.to_csv("data/synthetic_data.csv", index=False)  # Guardar los escenarios generados

print("Datos sintéticos generados y guardados en data/synthetic_data.csv")
