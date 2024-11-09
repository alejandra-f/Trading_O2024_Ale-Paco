import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

# 1. Función para descargar datos históricos
def download_stock_data(ticker, start_date, end_date, file_path="data/historical_data.csv"):
    data = yf.download(ticker, start=start_date, end=end_date)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path)
    print(f"Datos históricos de {ticker} guardados en {file_path}")
    return data

# 2. Construcción del modelo GAN

def build_generator(input_dim=100, output_dim=1):
    generator = Sequential([
        Input(shape=(input_dim,)),
        Dense(128),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Dense(output_dim, activation='tanh')
    ])
    return generator

def build_discriminator(input_shape=(1,)):
    discriminator = Sequential([
        Input(shape=input_shape),
        Dense(512),
        LeakyReLU(0.2),
        Dense(256),
        LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])
    return discriminator

# 3. Entrenamiento del GAN

def train_gan(generator, discriminator, gan, data, epochs=1000, batch_size=64):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]
        
        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_data = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)
        
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# 4. Estrategia de Trading

class TradingStrategy:
    def __init__(self, stop_loss=0.05, take_profit=0.1):
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def apply_strategy(self, data):
        operations = []
        position_open = False
        entry_price = None
        
        for i in range(1, len(data)):
            price = data["Close"].iloc[i]
            if not position_open:
                entry_price = price
                position_open = True
            else:
                change = (price - entry_price) / entry_price
                if change <= -self.stop_loss:
                    operations.append({"type": "sell", "price": price, "result": "stop_loss", "change": change})
                    position_open = False
                elif change >= self.take_profit:
                    operations.append({"type": "sell", "price": price, "result": "take_profit", "change": change})
                    position_open = False

        return pd.DataFrame(operations)

# 5. Función de Backtesting para probar múltiples niveles de stop-loss/take-profit

def backtest_strategy(data, stop_loss_levels, take_profit_levels):
    """
    Realiza el backtest de la estrategia en los datos con múltiples combinaciones de stop-loss y take-profit.
    
    Args:
        data (DataFrame): Datos de precios con columna 'Close'.
        stop_loss_levels (list): Lista de niveles de stop-loss a probar.
        take_profit_levels (list): Lista de niveles de take-profit a probar.
        
    Returns:
        DataFrame: Resultados del backtest para cada combinación de parámetros.
    """
    results = []
    
    for stop_loss in stop_loss_levels:
        for take_profit in take_profit_levels:
            strategy = TradingStrategy(stop_loss=stop_loss, take_profit=take_profit)
            operations = strategy.apply_strategy(data)
            metrics = calculate_metrics(operations)
            metrics["stop_loss"] = stop_loss
            metrics["take_profit"] = take_profit
            results.append(metrics)
    
    return pd.DataFrame(results)

# 6. Cálculo de Métricas

def calculate_metrics(operations):
    if operations.empty:
        return {
            "win_loss_ratio": 0,
            "total_operations": 0,
            "total_profits": 0,
            "total_losses": 0,
            "profit_loss_ratio": 0,
            "average_change": 0
        }

    profits = operations[operations["result"] == "take_profit"].shape[0]
    losses = operations[operations["result"] == "stop_loss"].shape[0]
    win_loss_ratio = profits / max(losses, 1)
    profit_loss_ratio = operations[operations["change"] > 0]["change"].mean() / abs(operations[operations["change"] < 0]["change"].mean())
    average_change = operations["change"].mean()
    
    metrics = {
        "win_loss_ratio": win_loss_ratio,
        "total_operations": len(operations),
        "total_profits": profits,
        "total_losses": losses,
        "profit_loss_ratio": profit_loss_ratio,
        "average_change": average_change
    }
    
    return metrics