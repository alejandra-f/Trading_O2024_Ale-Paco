import ta
import optuna
import pandas as pd
import numpy as np

from itertools import product
from ipywidgets import Datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sqlalchemy.dialects.mssql.information_schema import columns

# Variable global para almacenar el mejor resultado
global best_trial_info
best_trial_info = {'best_value': -float('inf'), 'best_combination': None, 'best_params': None}

# Cambia formato de fechas en el dataset
def update_datetime_format(data):
    try:
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
    except:
        try:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
        except:
            print('Error al cambiar formato de fecha')
    return data.copy()

# Backtesting para un conjunto de parámetros
def perform_backtesting(params, data):
    data = data.copy()
    update_datetime_format(data)

    # Parámetros optimizados
    tp_mult = params['tp_mult'] 
    sl_mult = params['sl_mult']  
    long_shares = params['long_shares']  
    short_shares = params['short_shares']  

    # Variables del backtesting
    cash = 1000000
    margin = 0
    comm_rate = 0.120 / 100
    margin_req = 0.30
    positions = []
    history = []
    portfolio_values = []

    for i, row in data.iterrows():
        # Margen requerido para posiciones cortas
        margin_needed = sum(pos["n_shares"] * row.Close * margin_req for pos in positions if pos["type"] == "short")

        # Transferir efectivo al margen si es necesario
        if margin_needed > margin:
            margin_diff = margin_needed - margin
            if cash >= margin_diff:
                cash -= margin_diff
                margin += margin_diff
            else:
                for pos in positions.copy():
                    if pos["type"] == "short":
                        profit = (pos["sold_at"] - row.Close) * pos["n_shares"]
                        cash += profit - (profit * comm_rate)
                        margin -= row.Close * pos["n_shares"] * margin_req
                        cash += pos["n_shares"] * row.Close * margin_req
                        history.append({"Datetime": row.Datetime, "operation": "closed short", "price": row.Close, "n_shares": pos["n_shares"]})
                        positions.remove(pos)
                        if sum(pos["n_shares"] * row.Close * margin_req for pos in positions if pos["type"] == "short") <= (margin + cash):
                            break

        # Revertir margen adicional
        if margin_needed < margin:
            excess_margin = margin - margin_needed
            cash += excess_margin
            margin -= excess_margin

        # Cerrar posiciones activas
        for pos in positions.copy():
            close_pos = False
            if pos["type"] == "long":
                if row.Close <= pos["stop_loss"] or row.Close >= pos["take_profit"]:
                    cash += row.Close * pos["n_shares"] * (1 - comm_rate)
                    close_pos = True
            elif pos["type"] == "short":
                if row.Close >= pos["stop_loss"] or row.Close <= pos["take_profit"]:
                    cash += (pos["sold_at"] - row.Close) * pos["n_shares"] * (1 - comm_rate)
                    margin -= pos["n_shares"] * row.Close * margin_req
                    cash += pos["n_shares"] * row.Close * margin_req
                    close_pos = True

            if close_pos:
                history.append({"Datetime": row.Datetime, "operation": f"closed {pos['type']}", "price": row.Close, "n_shares": pos["n_shares"]})
                positions.remove(pos)

        # Abrir nuevas posiciones basadas en señales
        # Larga
        if cash > row.Close * long_shares * (1 + comm_rate):
            if row.general_signals == 1:
                positions.append({
                    "Datetime": row.Datetime,
                    "bought_at": row.Close,
                    "type": "long",
                    "n_shares": long_shares,
                    "stop_loss": row.Close * sl_mult,
                    "take_profit": row.Close * tp_mult
                })
                cash -= row.Close * long_shares * (1 + comm_rate)
                history.append({"Datetime": row.Datetime, "operation": "opened long", "price": row.Close, "n_shares": long_shares})

        # Corta
        margin_short = row.Close * short_shares * margin_req
        if cash >= margin_short:
            if row.general_signals == -1:
                positions.append({
                    "Datetime": row.Datetime,
                    "sold_at": row.Close,
                    "type": "short",
                    "n_shares": short_shares,
                    "stop_loss": row.Close * sl_mult,
                    "take_profit": row.Close * tp_mult,
                })
                margin += margin_short
                cash -= margin_short
                history.append({"Datetime": row.Datetime, "operation": "opened short", "price": row.Close, "n_shares": short_shares})

        # Actualizar el valor del portafolio
        asset_value = sum([pos["n_shares"] * row.Close for pos in positions if pos["type"] == "long"])
        portfolio_values.append(cash + asset_value + margin)

    return portfolio_values[-1]

# Optimización de estrategias con Optuna
def optimize_strategy(trial, data):
    global best_trial_info

    data = data.copy()
    combinations = list(product([0, 1], repeat=5))
    combinations = [''.join(map(str, comb)) for comb in combinations][1:]

    for combination in combinations:
        # Parámetros a optimizar
        tp_mult = trial.suggest_float("tp_mult", 1.01, 1.1)
        sl_mult = trial.suggest_float("sl_mult", 0.9, 0.99)
        long_shares = trial.suggest_int("long_shares", 1, 100)
        short_shares = trial.suggest_int("short_shares", 1, 100)

        strategy_params = {
            "combination": combination,
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
            "long_shares": long_shares,
            "short_shares": short_shares
        }
        
        selected_strategies = []

        # Definir estrategias basadas en la combinación
        if combination[0] == '1':
            selected_strategies.append("RSI")
            strategy_params["RSI"] = {
                "window": trial.suggest_int("RSI_window", 1, 25)
            }
        if combination[1] == '1':
            selected_strategies.append("BB")
            strategy_params["BB"] = {
                "window": trial.suggest_int("BB_window", 1, 25),
                "window_dev": trial.suggest_float("BB_window_dev", 1.2, 2.5)
            }
        if combination[2] == '1':
            selected_strategies.append("WMA")
            strategy_params["WMA"] = {
                'window': trial.suggest_int("WMA_window", 1, 25)
            }
        if combination[3] == '1':
            selected_strategies.append("STO")
            strategy_params["STO"] = {
                "k_window": trial.suggest_int("k_window", 1, 25),
                "d_window": trial.suggest_int("d_window", 1, 15)
            }
        if combination[4] == '1':
            selected_strategies.append("DMI")
            strategy_params["DMI"] = {
                "window": trial.suggest_int("DMI_window", 1, 25)
            }

        # Generar señales basadas en las estrategias seleccionadas
        def generate_signals(selected_strategies, strategy_params, data):
            signal_columns = []

            if "RSI" in selected_strategies:
                window_rsi = strategy_params["RSI"]["window"]
                rsi_indicator = ta.momentum.RSIIndicator(data['Close'], window=window_rsi, fillna=False)
                data['RSI'] = rsi_indicator.rsi()

                data['RSI_signal'] = 0
                data.loc[data['Close'] > data['RSI'], 'RSI_signal'] = 1
                data.loc[data['Close'] < data['RSI'], 'RSI_signal'] = -1
                signal_columns.append('RSI_signal')

            if "BB" in selected_strategies:
                bb_params = strategy_params["BB"]
                window_bb = bb_params["window"]
                window_dev_bb = bb_params["window_dev"]

                bb_indicator = ta.volatility.BollingerBands(data['Close'], window=window_bb, window_dev=window_dev_bb, fillna=False)
                data['BB_hband'] = bb_indicator.bollinger_hband()
                data['BB_lband'] = bb_indicator.bollinger_lband()

                data['BB_signal'] = 0
                data.loc[data['Close'] < data['BB_lband'], 'BB_signal'] = 1
                data.loc[data['Close'] > data['BB_hband'], 'BB_signal'] = -1
                signal_columns.append('BB_signal')

            if "WMA" in selected_strategies:
                window_wma = strategy_params["WMA"]["window"]
                wma_indicator = ta.trend.WMAIndicator(data['Close'], window=window_wma, fillna=False)
                data['WMA'] = wma_indicator.wma()

                data['WMA_signal'] = 0
                data.loc[data['Close'] > data['WMA'], 'WMA_signal'] = 1
                data.loc[data['Close'] < data['WMA'], 'WMA_signal'] = -1
                signal_columns.append('WMA_signal')

            if "STO" in selected_strategies:
                sto_params = strategy_params["STO"]
                k_window_sto = sto_params["k_window"]
                d_window_sto = sto_params["d_window"]

                sr_indicator = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=k_window_sto, smooth_window=d_window_sto, fillna=False)
                data['SR_K'] = sr_indicator.stoch()
                data['SR_D'] = sr_indicator.stoch_signal()

                data['SR_signal'] = 0
                data.loc[(data['SR_K'] > data['SR_D']) & (data['SR_K'].shift(1) < data['SR_D'].shift(1)), 'SR_signal'] = 1
                data.loc[(data['SR_K'] < data['SR_D']) & (data['SR_K'].shift(1) > data['SR_D'].shift(1)), 'SR_signal'] = -1
                signal_columns.append('SR_signal')

            if "DMI" in selected_strategies:
                dmi_params = strategy_params["DMI"]
                window_dmi = dmi_params["window"]

                adx_indicator = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=window_dmi, fillna=False)
                data['ADX'] = adx_indicator.adx()
                data['DI+'] = adx_indicator.adx_pos()
                data['DI-'] = adx_indicator.adx_neg()

                data['DMI_signal'] = 0
                data.loc[(data['DI+'] > data['DI-']) & (data['ADX'] > 20), 'DMI_signal'] = 1
                data.loc[(data['DI+'] < data['DI-']) & (data['ADX'] > 20), 'DMI_signal'] = -1
                signal_columns.append('DMI_signal')

            data['general_signals'] = data[signal_columns].mean(axis=1, skipna=True)
            data.loc[data['general_signals'] > 0.4, 'general_signals'] = 1
            data.loc[(data['general_signals'] >= -0.4) & (data['general_signals'] <= 0.4), 'general_signals'] = 0
            data.loc[data['general_signals'] < -0.4, 'general_signals'] = -1

            return data

        # Generar señales y ejecutar backtesting
        data = generate_signals(selected_strategies=selected_strategies, strategy_params=strategy_params, data=data)
        portfolio_value = perform_backtesting(strategy_params, data)

        # Almacenar mejor resultado
        if portfolio_value > best_trial_info['best_value']:
            best_trial_info['best_value'] = portfolio_value
            best_trial_info['best_combination'] = combination
            best_trial_info['best_params'] = strategy_params.copy()

            trial.set_user_attr('best_combination', combination)

    return best_trial_info['best_value']

# Backtesting final
def final_backtesting(strategy_params, data):
    data = data.copy()

    # Parámetros
    tp_mult = strategy_params['tp_mult']
    sl_mult = strategy_params['sl_mult']
    long_shares = strategy_params['long_shares']
    short_shares = strategy_params['short_shares']

    # Variables
    cash = 10_000
    margin = 0
    positions = []
    history = []
    portfolio_values = []
    comm_rate = 0.125 / 100
    margin_req = 0.25

    # Iterar sobre filas de datos
    for i, row in data.iterrows():
        margin_needed = sum(pos["n_shares"] * row.Close * margin_req for pos in positions if pos["type"] == "short")

        if margin_needed > margin:
            margin_diff = margin_needed - margin
            if cash >= margin_diff:
                cash -= margin_diff
                margin += margin_diff
            else:
                for pos in positions.copy():
                    if pos["type"] == "short":
                        profit = (pos["sold_at"] - row.Close) * pos["n_shares"]
                        cash += profit - (profit * comm_rate)
                        margin -= row.Close * pos["n_shares"] * margin_req
                        cash += pos["n_shares"] * row.Close * margin_req
                        history.append({"Datetime": row.Datetime, "operation": "closed short", "price": row.Close, "n_shares": pos["n_shares"]})
                        positions.remove(pos)
                        if sum(pos["n_shares"] * row.Close * margin_req for pos in positions if pos["type"] == "short") <= (margin + cash):
                            break

        if margin_needed < margin:
            excess_margin = margin - margin_needed
            cash += excess_margin
            margin -= excess_margin

        for pos in positions.copy():
            close_pos = False
            if pos["type"] == "long":
                if row.Close <= pos["stop_loss"] or row.Close >= pos["take_profit"]:
                    cash += row.Close * pos["n_shares"] * (1 - comm_rate)
                    close_pos = True
            elif pos["type"] == "short":
                if row.Close >= pos["stop_loss"] or row.Close <= pos["take_profit"]:
                    cash += (pos["sold_at"] - row.Close) * pos["n_shares"] * (1 - comm_rate)
                    margin -= pos["n_shares"] * row.Close * margin_req
                    cash += pos["n_shares"] * row.Close * margin_req
                    close_pos = True

            if close_pos:
                history.append({"Datetime": row.Datetime, "operation": f"closed {pos['type']}", "price": row.Close, "n_shares": pos["n_shares"]})
                positions.remove(pos)

        if cash > row.Close * long_shares * (1 + comm_rate):
            if row.general_signals == 1:
                positions.append({
                    "Datetime": row.Datetime,
                    "bought_at": row.Close,
                    "type": "long",
                    "n_shares": long_shares,
                    "stop_loss": row.Close * sl_mult,
                    "take_profit": row.Close * tp_mult
                })
                cash -= row.Close * long_shares * (1 + comm_rate)
                history.append({"Datetime": row.Datetime, "operation": "opened long", "price": row.Close, "n_shares": long_shares})

        margin_short = row.Close * short_shares * margin_req
        if cash >= margin_short:
            if row.general_signals == -1:
                positions.append({
                    "Datetime": row.Datetime,
                    "sold_at": row.Close,
                    "type": "short",
                    "n_shares": short_shares,
                    "stop_loss": row.Close * sl_mult,
                    "take_profit": row.Close * tp_mult
                })
                margin += margin_short
                cash -= margin_short
                history.append({"Datetime": row.Datetime, "operation": "opened short", "price": row.Close, "n_shares": short_shares})

        asset_value = sum([pos["n_shares"] * row.Close for pos in positions if pos["type"] == "long"])
        portfolio_values.append(cash + asset_value + margin)

    return portfolio_values[-1], portfolio_values

def add_strategy_params(combination, data, strategy_params):

    data = data.copy()
    selected_strategies = []

    if combination[0] == '1':
        selected_strategies.append("RSI")
        
    if combination[1] == '1':
        selected_strategies.append("BB")

    if combination[2] == '1':
        selected_strategies.append("WMA")

    if combination[3] == '1':
        selected_strategies.append("STO")

    if combination[4] == '1':
        selected_strategies.append("DMI")

    # Generar señales según las estrategias seleccionadas
    def generate_signals(selected_strategies=selected_strategies, strategy_params=strategy_params, data=data):
        
        signal_columns = []
        
        # RSI
        if "RSI" in selected_strategies:
            window_rsi = strategy_params["RSI_window"]
            
            rsi_indicator = ta.momentum.RSIIndicator(data['Close'], window=window_rsi, fillna=False)
            data['RSI'] = rsi_indicator.rsi()

            # Señales de RSI
            data['RSI_signal'] = 0
            data.loc[data['Close'] > data['RSI'], 'RSI_signal'] = 1  # Señal de compra
            data.loc[data['Close'] < data['RSI'], 'RSI_signal'] = -1  # Señal de venta
            signal_columns.append('RSI_signal')

        # BB
        if "BB" in selected_strategies:
            window_bb = strategy_params["BB_window"]
            window_bb_dev = strategy_params["BB_window_dev"]

            bb_indicator = ta.volatility.BollingerBands(data['Close'], window=window_bb, window_dev=window_bb_dev, fillna=False)
            data['BB_hband'] = bb_indicator.bollinger_hband()
            data['BB_lband'] = bb_indicator.bollinger_lband()

            # Señales de BB
            data['BB_signal'] = 0
            data.loc[data['Close'] < data['BB_lband'], 'BB_signal'] = 1  # Señal de compra
            data.loc[data['Close'] > data['BB_hband'], 'BB_signal'] = -1  # Señal de venta
            signal_columns.append('BB_signal')

        # WMA
        if "WMA" in selected_strategies:
            window_wma = strategy_params["WMA_window"]
            
            wma_indicator = ta.trend.WMAIndicator(data['Close'], window=window_wma, fillna=False)
            data['WMA'] = wma_indicator.wma()

            # Señales de WMA
            data['WMA_signal'] = 0
            data.loc[data['Close'] > data['WMA'], 'WMA_signal'] = 1  # Señal de compra
            data.loc[data['Close'] < data['WMA'], 'WMA_signal'] = -1  # Señal de venta
            signal_columns.append('WMA_signal')

        # STO
        if "STO" in selected_strategies:
            k_window_sto = strategy_params["k_window"]
            d_window_sto = strategy_params["d_window"]

            sto_indicator = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=k_window_sto, smooth_window=d_window_sto, fillna=False)
            data['STO_K'] = sto_indicator.stoch()
            data['STO_D'] = sto_indicator.stoch_signal()

            # Señales de STO
            data['STO_signal'] = 0
            data.loc[(data['STO_K'] > data['STO_D']) & (data['STO_K'].shift(1) < data['STO_D'].shift(1)), 'STO_signal'] = 1  # Señal de compra
            data.loc[(data['STO_K'] < data['STO_D']) & (data['STO_K'].shift(1) > data['STO_D'].shift(1)), 'STO_signal'] = -1  # Señal de venta
            signal_columns.append('STO_signal')

        # DMI
        if "DMI" in selected_strategies:
            window_dmi = strategy_params["DMI_window"]
            adx_threshold = strategy_params["ADX_threshold"]
            
            dmi_indicator = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=window_dmi, fillna=False)
            data['ADX'] = dmi_indicator.adx()
            data['DI+'] = dmi_indicator.adx_pos()
            data['DI-'] = dmi_indicator.adx_neg()

            # Señales de DMI
            data['DMI_signal'] = 0
            data.loc[(data['DI+'] > data['DI-']) & (data['ADX'] > adx_threshold), 'DMI_signal'] = 1  # Señal de compra
            data.loc[(data['DI+'] < data['DI-']) & (data['ADX'] > adx_threshold), 'DMI_signal'] = -1  # Señal de venta
            signal_columns.append('DMI_signal')

        # Combinación de señales
        data['general_signals'] = data[signal_columns].mean(axis=1, skipna=True)

        # Definir señales generales basadas en umbrales
        data.loc[data['general_signals'] > 0.4, 'general_signals'] = 1
        data.loc[(data['general_signals'] >= -0.4) & (data['general_signals'] <= 0.4), 'general_signals'] = 0
        data.loc[data['general_signals'] < -0.4, 'general_signals'] = -1

        return data

    # Generar señales con las estrategias
    data = generate_signals(selected_strategies=selected_strategies, strategy_params=strategy_params, data=data)

    return data
