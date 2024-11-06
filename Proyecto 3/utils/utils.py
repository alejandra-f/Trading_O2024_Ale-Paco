import yfinance as yf
import pandas as pd
import numpy as np

def download_data(ticker, start_date, end_date):
    """
    Descarga datos históricos de Yahoo Finance para el ticker dado.
    Calcula señales de cruce de medias móviles (SMA 20 y SMA 50).
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Generar señales basadas en el cruce de medias móviles
    data['Signal'] = 0
    data['Signal'][data['SMA_20'] > data['SMA_50']] = 1  # Señal de compra
    data['Signal'][data['SMA_20'] < data['SMA_50']] = -1  # Señal de venta

    # Guardar los datos con señales en un archivo CSV para referencia futura
    data.to_csv('data/historical_data.csv')
    return data

def calculate_metrics(data, returns):
    """
    Calcula el Sharpe Ratio y el Drawdown Máximo.
    """
    # Sharpe Ratio con una tasa libre de riesgo de 0
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Suponiendo 252 días de trading
    
    # Drawdown máximo
    cumulative = data['Close'].cummax()
    max_drawdown = (cumulative - data['Close']).max() / cumulative.max()

    return {'Sharpe Ratio': sharpe_ratio, 'Max Drawdown': max_drawdown}