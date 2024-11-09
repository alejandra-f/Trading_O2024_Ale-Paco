import pandas as pd
from utils.utils import TradingStrategy, calculate_metrics

# Clase para realizar el backtest
class Backtest:
    def __init__(self, data, stop_loss, take_profit):
        self.data = data
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy = TradingStrategy(stop_loss, take_profit)

    def run_backtest(self):
        results = self.strategy.apply_strategy(self.data)
        metrics = calculate_metrics(results)  # Calcula las métricas
        return results, metrics

# Ejecución del backtest en datos reales y sintéticos
historical_data = pd.read_csv("data/historical_data.csv")
synthetic_data = pd.read_csv("data/synthetic_data.csv")

backtests = []
for i, scenario in enumerate(synthetic_data["scenarios"]):
    bt = Backtest(scenario, stop_loss=0.05, take_profit=0.1)
    results, metrics = bt.run_backtest()
    backtests.append(metrics)

# Guardar los resultados
results_df = pd.DataFrame(backtests)
results_df.to_csv("data/backtest_results.csv", index=False)
print("Resultados del backtest guardados en data/backtest_results.csv")