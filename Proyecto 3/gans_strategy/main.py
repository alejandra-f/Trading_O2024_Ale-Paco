import pandas as pd
import numpy as np
from utils.utils import download_data, calculate_metrics, calculate_additional_metrics
from gans_strategy.backtest import Backtest
from gans_strategy.model import train_gan, generate_scenarios

# Parámetros
ticker = "AAPL"
start_date = "2013-01-01"
end_date = "2023-01-01"
n_scenarios = 100
stop_loss_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
take_profit_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Paso 1: Descargar y preparar datos históricos
data = download_data(ticker, start_date, end_date)

# Paso 2: Entrenar y generar escenarios con GAN
gan_generator = train_gan(data)  # Entrenar el modelo GAN
scenarios = generate_scenarios(gan_generator, n_scenarios=n_scenarios)  # Generar 100 escenarios
scenarios.to_csv('data/generated_scenarios.csv')  # Guardar los escenarios generados

# Paso 3: Realizar backtesting en el dataset real con diferentes niveles de stop-loss/take-profit
real_results = []
for sl in stop_loss_levels:
    for tp in take_profit_levels:
        backtest = Backtest(data, stop_loss=sl, take_profit=tp)
        final_balance = backtest.run()
        history = backtest.get_history()
        valid_indices = [idx for idx in history.index if idx in data.index]
        if valid_indices:
            filtered_history = history.loc[valid_indices]
            returns = pd.Series([entry[2] for entry in filtered_history.values]).pct_change().dropna()
            metrics = calculate_metrics(data, returns)
            additional_metrics = calculate_additional_metrics(data, returns)
            real_results.append({
                'stop_loss': sl,
                'take_profit': tp,
                'final_balance': final_balance,
                'metrics': metrics,
                'additional_metrics': additional_metrics
            })
real_results_df = pd.DataFrame(real_results)
real_results_df.to_csv('data/backtest_real_results.csv', index=False)

# Paso 4: Realizar backtesting en los escenarios generados
scenario_results = []
for i, scenario in enumerate(scenarios):
    for sl in stop_loss_levels:
        for tp in take_profit_levels:
            backtest = Backtest(scenario, stop_loss=sl, take_profit=tp)
            final_balance = backtest.run()
            history = backtest.get_history()
            valid_indices = [idx for idx in history.index if idx in scenario.index]
            if valid_indices:
                filtered_history = history.loc[valid_indices]
                returns = pd.Series([entry[2] for entry in filtered_history.values]).pct_change().dropna()
                metrics = calculate_metrics(scenario, returns)
                additional_metrics = calculate_additional_metrics(scenario, returns)
                scenario_results.append({
                    'scenario': i+1,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'final_balance': final_balance,
                    'metrics': metrics,
                    'additional_metrics': additional_metrics
                })
scenario_results_df = pd.DataFrame(scenario_results)
scenario_results_df.to_csv('data/backtest_scenario_results.csv', index=False)

# Paso 5: Crear estrategia pasiva (comprar y mantener)
passive_backtest = Backtest(data, passive_strategy=True)
passive_final_balance = passive_backtest.run()
passive_history = passive_backtest.get_history()
valid_indices = [idx for idx in passive_history.index if idx in data.index]
if valid_indices:
    filtered_history = passive_history.loc[valid_indices]
    passive_returns = pd.Series([entry[2] for entry in filtered_history.values]).pct_change().dropna()
    passive_metrics = calculate_metrics(data, passive_returns)
    passive_additional_metrics = calculate_additional_metrics(data, passive_returns)
    passive_results = {
        'final_balance': passive_final_balance,
        'metrics': passive_metrics,
        'additional_metrics': passive_additional_metrics
    }
    passive_results_df = pd.DataFrame([passive_results])
    passive_results_df.to_csv('data/backtest_passive_results.csv', index=False)

print("Pipeline completado")


