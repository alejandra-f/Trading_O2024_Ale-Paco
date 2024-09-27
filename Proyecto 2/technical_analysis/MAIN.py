import sys 
import os
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import logging

# Ajustar ruta del directorio
ruta_directorio_superior = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ruta_directorio_superior)

# Importar las funciones de utils, usando los nuevos nombres
from utils.UTILS import perform_backtesting, optimize_strategy, update_datetime_format, add_strategy_params, final_backtesting

# Cambiar directorio actual
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

# Cargar los datos, enfocándonos solo en el dataset de 5 minutos
data_5min_T = pd.read_csv("data/aapl_5m_train.csv")
data_5min_V = pd.read_csv("data/aapl_5m_test.csv")

# Aplicar la función de actualización de formato de fecha a los datasets
data_5min_V = update_single_datetime_format(data_5min_V)
data_5min_T = update_single_datetime_format(data_5min_T)

# Configurar el logger para Optuna
logging.getLogger("optuna").setLevel(logging.ERROR)

# Crear el estudio de Optuna y optimizar
study = optuna.create_study(direction='maximize')

# Optimización usando el dataset de entrenamiento de 5 minutos
study.optimize(lambda trial: objective(trial, data=data_5min_T), n_trials=5, show_progress_bar=True)

# Resultados de la optimización
print("Mejor valor del portafolio:", study.best_value)
params = study.best_params
print("Mejores parámetros:", study.best_params)

# Obtener los detalles del mejor trial
best_trial = study.best_trial
best_com = best_trial.user_attrs.get('best_combination')
print("Mejor combinación de indicadores:", best_com)

# Aplicar los parámetros y obtener las señales en el dataset de validación
data_5min_V = add_strategy_params(best_com, data_5min_V, params)

# Aplicar nuevamente la actualización de formato de fecha (si fuera necesario)
data_5min_V = update_single_datetime_format(data_5min_V)

# Realizar backtesting final con el dataset de validación de 5 minutos
final_portfolio_value, portfolio_values = final_backtesting(params, data_5min_V)

# Crear DataFrame combinado con los valores del portafolio
combined_df = pd.DataFrame({
        'Datetime': data_5min_V['Datetime'],
        'Close': data_5min_V['Close'],
        'PortfolioValue': portfolio_values
})

# Ruta del archivo CSV donde quieres guardar los datos
file_path = r"C:\Users\aleef\OneDrive\Documentos\TRADING\Trading\Proyecto 2\data"

# Guardar el DataFrame en un archivo CSV
combined_df.to_csv(file_path, index=False)

print(f"Resultados guardados en {file_path}")
