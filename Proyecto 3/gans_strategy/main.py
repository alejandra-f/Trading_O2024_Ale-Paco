import pandas as pd
from utils.utils import download_data, calculate_metrics
from gans_strategy.backtest import Backtest
from gans_strategy.model import train_gan, generate_scenarios

# Parámetros
ticker = "AAPL"
start_date = "2013-01-01"
end_date = "2023-01-01"

# Paso 1: Descargar y preparar datos históricos
data = download_data(ticker, start_date, end_date)

# Paso 2: Entrenar y generar escenarios con GAN
# Supongamos que `train_gan` devuelve un modelo generador entrenado
gan_generator = train_gan(data)  # Entrenar el modelo GAN
scenarios = generate_scenarios(gan_generator, n_scenarios=100)  # Generar 100 escenarios
scenarios.to_csv('data/generated_scenarios.csv')  # Guardar los escenarios generados

# Paso 3: Realizar backtesting en el dataset real
backtest = Backtest(data)
final_balance = backtest.run()
print(f"Saldo final después del backtest en datos reales: {final_balance}")

# Obtener el historial de operaciones y calcular métricas
history = backtest.get_history()
history.to_csv('data/backtest_results.csv', index=False)  # Guardar resultados del backtest
returns = pd.Series([entry[2] for entry in history.values]).pct_change().dropna()
metrics = calculate_metrics(data, returns)
print("Métricas en datos reales:", metrics)

# Paso 4: Realizar backtesting en los escenarios generados
for i, scenario in enumerate(scenarios):
    backtest = Backtest(scenario)
    final_balance = backtest.run()
    print(f"Saldo final en escenario generado {i+1}: {final_balance}")

print("Pipeline completado")