import pandas as pd
import numpy as np
import ta
import optuna
import itertools
from ipywidgets import Datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product

from sqlalchemy.dialects.mssql.information_schema import columns

global best_trial_info
best_trial_info={'best_value': -float('inf'), 'best_combination': None, 'best_params':None}

def change_dateformat(datasets:list):
    for data in datasets:
        try:
            data.rename(columns={'Date': 'Datetime'},inplace=True)
        except:
            try:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
            except:
                continue

def change_dateformat_data(data):
    try:
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
    except:
        try:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
        except:
            print('a')
    data = data.copy()
    return data

#Backtesting

def objective_backtesting(params, data):
    # Copia de los datos para evitar modificar el original
    data = data.copy()
    change_dateformat_data(data)

    #Definir hiperparámetros a optimizar
    take_profit_multiplier=params['take_profit_multiplier']
    stop_loss_multiplier = params['stop_loss_multiplier']
    n_shares_long = params['n_shares_long']
    n_shares_short = params['n_shares_short']

    # Inicializar variables para el backtesting
    cash = 1_000_000
    margin_account = 0  # Cuenta de margen para operaciones en corto
    active_operations = []
    history = []
    portfolio_value = []
    commission = 0.125 / 100
    margin_call = 0.25  # Porcentaje de margen requerido

    for i, row in data.iterrows():
        # Actualizar el margen necesario para las posiciones cortas
        margin_required = sum(op["n_shares"] * row.Close * margin_call
                              for op in active_operations if op["type"] == "short")

        if margin_required > margin_account:
            additional_margin_needed = margin_required - margin_account

            if cash >= additional_margin_needed:
                # Si hay suficiente efectivo, transferirlo a la cuenta de margen
                cash -= additional_margin_needed
                margin_account += additional_margin_needed
            else:
                # Si no hay suficiente efectivo, cerrar posiciones cortas hasta que el margen sea suficiente
                for operation in active_operations.copy():
                    if operation["type"] == "short":
                        profit = (operation["sold_at"] - row.Close) * operation["n_shares"]
                        cash += profit - (profit * commission)  # Ajustar por comisión
                        margin_account -= row.Close * operation["n_shares"] * margin_call  # Liberar el margen reservado
                        cash+= operation["n_shares"] * row.Close * margin_call
                        history.append({"Datetime": row.Datetime, "operation": "closed short", "price": row.Close, "n_shares": operation["n_shares"]})
                        active_operations.remove(operation)
                        if sum(op["n_shares"] * row.Close * margin_call for op in active_operations if op["type"] == "short") <= (margin_account+cash):
                            break  # Salir del bucle si el margen es suficiente


        if margin_required < margin_account:
            excess_margin = margin_account - margin_required
            cash += excess_margin
            margin_account -= excess_margin

        # Cerrar operaciones largas y cortas según stop loss y take profit
        for operation in active_operations.copy():
            close_position = False
            if operation["type"] == "long":
                if row.Close <= operation["stop_loss"] or row.Close >= operation["take_profit"]:
                    cash += row.Close * operation["n_shares"] * (1 - commission)
                    close_position = True
            elif operation["type"] == "short":
                if row.Close >= operation["stop_loss"] or row.Close <= operation["take_profit"]:
                    cash += (operation["sold_at"] - row.Close) * operation["n_shares"] * (1 - commission)
                    margin_account -= operation["n_shares"] * row.Close * margin_call
                    cash += operation["n_shares"] * row.Close * margin_call
                    close_position = True

            if close_position:
                history.append({"Datetime": row.Datetime, "operation": f"closed {operation['type']}", "price": row.Close, "n_shares": operation["n_shares"]})
                active_operations.remove(operation)

        # Abrir nuevas operaciones según las señales

        # Long
        if cash > row.Close * n_shares_long * (1 + commission):
            if row.general_signals == 1:
                active_operations.append({
                    "Datetime": row.Datetime,
                    "bought_at": row.Close,
                    "type": "long",
                    "n_shares": n_shares_long,
                    "stop_loss": row.Close * stop_loss_multiplier,
                    "take_profit": row.Close * take_profit_multiplier
                })
                cash -= row.Close * n_shares_long * (1 + commission)
                history.append({"Datetime": row.Datetime, "operation": "opened long", "price": row.Close, "n_shares": n_shares_long})

        # Short
        required_margin_for_new_short = row.Close * n_shares_short * margin_call
        if cash >= required_margin_for_new_short:  # Verificar si hay suficiente efectivo para el margen requerido
            if row.general_signals == -1:  # Ejemplo de señal para operación corta
                active_operations.append({
                    "Datetime": row.Datetime,
                    "sold_at": row.Close,
                    "type": "short",
                    "n_shares": n_shares_short,
                    "stop_loss": row.Close * stop_loss_multiplier,
                    "take_profit": row.Close * take_profit_multiplier,
                })
                margin_account += required_margin_for_new_short
                cash -= required_margin_for_new_short  # Reservar efectivo para el margen
                history.append({"Datetime": row.Datetime, "operation": "opened short", "price": row.Close, "n_shares": n_shares_short})

        # Actualizar el valor de la cartera
        asset_vals = sum([op["n_shares"] * row.Close for op in active_operations if op["type"] == "long"])
        portfolio_value.append(cash + asset_vals + margin_account)

    final_portfolio_value = portfolio_value[-1]
    return final_portfolio_value

# Objective Functions
def objective(trial, data):
    global best_trial_info

    data=data.copy()
    combinaciones = list(itertools.product([0, 1], repeat=5))
    combinations=[''.join(map(str, com)) for com in combinaciones]
    combinations = combinations[1:]

    best_portfolio_value = -float('inf')
    best_combination = None
    best_params={}

    for com in combinations:
        # Hiperparámetros a optimizar
        take_profit_multiplier = trial.suggest_float("take_profit_multiplier", 1.01, 1.1)
        stop_loss_multiplier = trial.suggest_float("stop_loss_multiplier", 0.9, 0.99)
        n_shares_long = trial.suggest_int("n_shares_long", 1, 100)
        n_shares_short = trial.suggest_int("n_shares_short", 1, 100)

        # Se van a or agregando los params de los indicadores
        params = {
            "com": com,
            "take_profit_multiplier": trial.suggest_float("take_profit_multiplier", 1.01, 1.1),
            "stop_loss_multiplier": trial.suggest_float("stop_loss_multiplier", 0.9, 0.99),
            "n_shares_long": trial.suggest_int("n_shares_long", 1, 100),
            "n_shares_short": trial.suggest_int("n_shares_short", 1, 100),
        }
        strats = []

        if com[0] == '1':
            strats.append("RSI")
            params["RSI"] ={
                "window": trial.suggest_int("RSI_window", 1, 30),
                "up_threshold": trial.suggest_int("RSI_up_threshold", 50, 80),
                "lw_threshold": trial.suggest_int("RSI_lw_threshold", 30, 50)
                }
        if com[1] == '1':
            strats.append("BB") # Bollinger Bands
            params["BB"]={
                "window": trial.suggest_int("BB_window", 1, 30),
                "window_dev": trial.suggest_float("BB_window_dev", 1.0, 3.0)
            }

        if com[2] == '1':
            strats.append("WMA") # Weighted Moving Average
            params["WMA"]={
                'window': trial.suggest_int("WMA_window",1,30)
            }

        if com[3] == '1':
            strats.append("STO") # Stochastic Oscillator
            params["STO"]={
                "k_window": trial.suggest_int("k_window", 1, 30),
                "d_window": trial.suggest_int("d_window", 1, 20)
            }

        if com[4] == '1':
            strats.append("DMI") #Indice de Movimiento Direccional
            params["DMI"] ={
                "window": trial.suggest_int("DMI_window", 1, 30)
            }

        def signal_functions(strats=strats, params=params, data=data):

            # signals = []
            signal_columns = []
            # n_strats = len(strats)

            if "RSI" in strats:
                rsi_params = params["RSI"]
                window_rsi = rsi_params["window"]
                up_threshold_rsi = rsi_params["up_threshold"]
                lw_threshold_rsi = rsi_params["lw_threshold"]

                rsi_indicator = ta.momentum.RSIIndicator(data['Close'], window=window_rsi, fillna=False)
                data['RSI'] = rsi_indicator.rsi()

                # Definición de señales
                data['RSI_signal'] = 0
                data.loc[data['RSI'] < up_threshold_rsi, 'RSI_signal'] = 1  # Señal de compra
                data.loc[data['RSI'] > lw_threshold_rsi, 'RSI_signal'] = -1  # Señal de venta
                signal_columns.append('RSI_signal')
                # signals.append(data['RSI_signal'])

            if "BB" in strats:
                bb_params = params["BB"]
                window_bb = bb_params["window"]
                window_bb_dev = bb_params["window_dev"]

                bb_indicator = ta.volatility.BollingerBands(data['Close'], window=window_bb, window_dev=window_bb_dev, fillna=False)
                data['BB_hband'] = bb_indicator.bollinger_hband()
                data['BB_lband'] = bb_indicator.bollinger_lband()

                # Define la señal de compra y venta basada en las Bandas de Bollinger (por ejemplo, cruce de precio y banda)
                data['BB_signal'] = 0
                data.loc[data['Close'] < data['BB_lband'], 'BB_signal'] = 1  # Señal de compra
                data.loc[data['Close'] > data['BB_hband'], 'BB_signal'] = -1  # Señal de venta
                signal_columns.append('BB_signal')

            if "WMA" in strats:
                wma_params = params["WMA"]
                window_wma = wma_params["window"]

                wma_indicator = ta.trend.WMAIndicator(data['Close'], window=window_wma, fillna=False)
                data['WMA'] = wma_indicator.wma()

                # Define la señal de compra y venta basada en el WMA (por ejemplo, cruzando la señal WMA)
                data['WMA_signal'] = 0
                data.loc[data['Close'] > data['WMA'], 'WMA_signal'] = 1  # Señal de compra
                data.loc[data['Close'] < data['WMA'], 'WMA_signal'] = -1  # Señal de venta
                signal_columns.append('WMA_signal')

            if "STO" in strats:
                sto_params = params["STO"]
                k_window_sto = sto_params["k_window"]
                d_window_sto = sto_params["d_window"]

                sr_indicator = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=k_window_sto, smooth_window=d_window_sto, fillna=False)
                data['SR_K'] = sr_indicator.stoch()
                data['SR_D'] = sr_indicator.stoch_signal()

                # Define la señal de compra y venta basada en el Stochastic Oscillator (por ejemplo, cruce de %K y %D)
                data['SR_signal'] = 0
                data.loc[(data['SR_K'] > data['SR_D']) & (data['SR_K'].shift(1) < data['SR_D'].shift(1)), 'SR_signal'] = 1  # Señal de compra
                data.loc[(data['SR_K'] < data['SR_D']) & (data['SR_K'].shift(1) > data['SR_D'].shift(1)), 'SR_signal'] = -1  # Señal de venta
                signal_columns.append('SR_signal')

            if "DMI" in strats:
                dmi_params=params["DMI"]
                window_dmi = dmi_params["window"]

                adx_indicator = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=window_dmi, fillna=False)
                data['ADX'] = adx_indicator.adx()
                data['DI+'] = adx_indicator.adx_pos()
                data['DI-'] = adx_indicator.adx_neg()

                # Define la señal de compra y venta basada en el ADX y DI+/DI-
                data['DMI_signal'] = 0
                data.loc[(data['DI+'] > data['DI-']) & (data['ADX'] > 20), 'DMI_signal'] = 1  # Señal de compra
                data.loc[(data['DI+'] < data['DI-']) & (data['ADX'] > 20), 'DMI_signal'] = -1  # Señal de venta
                signal_columns.append('DMI_signal')

            data['general_signals'] = data[signal_columns].mean(axis=1, skipna=True)

            # Asignar valores en la nueva columna según las condiciones
            data.loc[data['general_signals'] > 0.4, 'general_signals'] = 1
            data.loc[(data['general_signals'] >= -0.4) & (data['general_signals'] <= 0.4), 'general_signals'] = 0
            data.loc[data['general_signals'] < -0.4, 'general_signals'] = -1
            # Si no hay columnas presentes, asignar NaN a la nueva columna

            return data

        data = signal_functions(strats=strats, params=params, data=data)
        # data.dropna(inplace=True)

        portfolio_value = objective_backtesting(params, data)

        if portfolio_value > best_trial_info['best_value']:
            best_trial_info['best_value'] = portfolio_value
            best_trial_info['best_combination'] = com
            best_trial_info['best_params'] = params.copy()

            # Almacenar información adicional en user_attrs
            trial.set_user_attr('best_combination', com)

    return best_trial_info['best_value']


def backtesting_final(params, data):
    # Copia de los datos para evitar modificar el original
    data = data.copy()

    # Definir hiperparámetros a optimizar
    take_profit_multiplier = params['take_profit_multiplier']
    stop_loss_multiplier = params['stop_loss_multiplier']
    n_shares_long = params['n_shares_long']
    n_shares_short = params['n_shares_short']

    # Inicializar variables para el backtesting
    cash = 1_000_000
    margin_account = 0  # Cuenta de margen para operaciones en corto
    active_operations = []
    history = []
    portfolio_value = []
    commission = 0.125 / 100
    margin_call = 0.25  # Porcentaje de margen requerido

    for i, row in data.iterrows():
        # Actualizar el margen necesario para las posiciones cortas
        margin_required = sum(op["n_shares"] * row.Close * margin_call
                              for op in active_operations if op["type"] == "short")

        if margin_required > margin_account:
            additional_margin_needed = margin_required - margin_account

            if cash >= additional_margin_needed:
                # Si hay suficiente efectivo, transferirlo a la cuenta de margen
                cash -= additional_margin_needed
                margin_account += additional_margin_needed
            else:
                # Si no hay suficiente efectivo, cerrar posiciones cortas hasta que el margen sea suficiente
                for operation in active_operations.copy():
                    if operation["type"] == "short":
                        profit = (operation["sold_at"] - row.Close) * operation["n_shares"]
                        cash += profit - (profit * commission)  # Ajustar por comisión
                        margin_account -= row.Close * operation["n_shares"] * margin_call  # Liberar el margen reservado
                        cash += operation["n_shares"] * row.Close * margin_call
                        history.append({"Datetime": row.Datetime, "operation": "closed short", "price": row.Close,
                                        "n_shares": operation["n_shares"]})
                        active_operations.remove(operation)
                        if sum(op["n_shares"] * row.Close * margin_call for op in active_operations if
                               op["type"] == "short") <= (margin_account + cash):
                            break  # Salir del bucle si el margen es suficiente

        if margin_required < margin_account:
            excess_margin = margin_account - margin_required
            cash += excess_margin
            margin_account -= excess_margin

        # Cerrar operaciones largas y cortas según stop loss y take profit
        for operation in active_operations.copy():
            close_position = False
            if operation["type"] == "long":
                if row.Close <= operation["stop_loss"] or row.Close >= operation["take_profit"]:
                    cash += row.Close * operation["n_shares"] * (1 - commission)
                    close_position = True
            elif operation["type"] == "short":
                if row.Close >= operation["stop_loss"] or row.Close <= operation["take_profit"]:
                    cash += (operation["sold_at"] - row.Close) * operation["n_shares"] * (1 - commission)
                    margin_account -= operation["n_shares"] * row.Close * margin_call
                    cash += operation["n_shares"] * row.Close * margin_call
                    close_position = True

            if close_position:
                history.append(
                    {"Datetime": row.Datetime, "operation": f"closed {operation['type']}", "price": row.Close,
                     "n_shares": operation["n_shares"]})
                active_operations.remove(operation)

        # Abrir nuevas operaciones según las señales

        # Long
        if cash > row.Close * n_shares_long * (1 + commission):
            if row.general_signals == 1:
                active_operations.append({
                    "Datetime": row.Datetime,
                    "bought_at": row.Close,
                    "type": "long",
                    "n_shares": n_shares_long,
                    "stop_loss": row.Close * stop_loss_multiplier,
                    "take_profit": row.Close * take_profit_multiplier
                })
                cash -= row.Close * n_shares_long * (1 + commission)
                history.append({"Datetime": row.Datetime, "operation": "opened long", "price": row.Close,
                                "n_shares": n_shares_long})

        # Short
        required_margin_for_new_short = row.Close * n_shares_short * margin_call
        if cash >= required_margin_for_new_short:  # Verificar si hay suficiente efectivo para el margen requerido
            if row.general_signals == -1:  # Ejemplo de señal para operación corta
                active_operations.append({
                    "Datetime": row.Datetime,
                    "sold_at": row.Close,
                    "type": "short",
                    "n_shares": n_shares_short,
                    "stop_loss": row.Close * stop_loss_multiplier,
                    "take_profit": row.Close * take_profit_multiplier,
                })
                margin_account += required_margin_for_new_short
                cash -= required_margin_for_new_short  # Reservar efectivo para el margen
                history.append({"Datetime": row.Datetime, "operation": "opened short", "price": row.Close,
                                "n_shares": n_shares_short})

        # Actualizar el valor de la cartera
        asset_vals = sum([op["n_shares"] * row.Close for op in active_operations if op["type"] == "long"])
        portfolio_value.append(cash + asset_vals + margin_account)

    final_portfolio_value = portfolio_value[-1]
    return final_portfolio_value, portfolio_value


def add_params_data(com, data, params):
    data = data.copy()  # Asegúrate de que 'data' esté definida aquí o sea accesible como variable global
    strats = []

    if com[0] == '1':
        strats.append("RSI")

    if com[1] == '1':
        strats.append("BB")

    if com[2] == '1':
        strats.append("WMA")

    if com[3] == '1':
        strats.append("STO")

    if com[4] == '1':
        strats.append("DMI")

    def signal_functions(strats=strats, params=params, data=data):

        signal_columns = []

        if "RSI" in strats:
            window_rsi = params["RSI_window"]
            up_threshold_rsi = params["RSI_up_threshold"]
            lw_threshold_rsi = params["RSI_lw_threshold"]

            rsi_indicator = ta.momentum.RSIIndicator(data['Close'], window=window_rsi, fillna=False)
            data['RSI'] = rsi_indicator.rsi()

            # Definición de señales
            data['RSI_signal'] = 0
            data.loc[data['RSI'] < up_threshold_rsi, 'RSI_signal'] = 1  # Señal de compra
            data.loc[data['RSI'] > lw_threshold_rsi, 'RSI_signal'] = -1  # Señal de venta
            signal_columns.append('RSI_signal')
            # signals.append(data['RSI_signal'])

        if "BB" in strats:
            window_bb = params["BB_window"]
            window_bb_dev = params["BB_window_dev"]

            bb_indicator = ta.volatility.BollingerBands(data['Close'], window=window_bb, window_dev=window_bb_dev,
                                                        fillna=False)
            data['BB_hband'] = bb_indicator.bollinger_hband()
            data['BB_lband'] = bb_indicator.bollinger_lband()

            # Define la señal de compra y venta basada en las Bandas de Bollinger (por ejemplo, cruce de precio y banda)
            data['BB_signal'] = 0
            data.loc[data['Close'] < data['BB_lband'], 'BB_signal'] = 1  # Señal de compra
            data.loc[data['Close'] > data['BB_hband'], 'BB_signal'] = -1  # Señal de venta
            signal_columns.append('BB_signal')

        if "WMA" in strats:
            window_wma = params["WMA_window"]

            wma_indicator = ta.trend.WMAIndicator(data['Close'], window=window_wma, fillna=False)
            data['WMA'] = wma_indicator.wma()

            # Define la señal de compra y venta basada en el WMA (por ejemplo, cruzando la señal WMA)
            data['WMA_signal'] = 0
            data.loc[data['Close'] > data['WMA'], 'WMA_signal'] = 1  # Señal de compra
            data.loc[data['Close'] < data['WMA'], 'WMA_signal'] = -1  # Señal de venta
            signal_columns.append('WMA_signal')

        if "STO" in strats:
            k_window_sto = params["k_window"]
            d_window_sto = params["d_window"]

            sr_indicator = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'],
                                                            window=k_window_sto, smooth_window=d_window_sto,
                                                            fillna=False)
            data['SR_K'] = sr_indicator.stoch()
            data['SR_D'] = sr_indicator.stoch_signal()

            # Define la señal de compra y venta basada en el Stochastic Oscillator (por ejemplo, cruce de %K y %D)
            data['SR_signal'] = 0
            data.loc[(data['SR_K'] > data['SR_D']) & (
                        data['SR_K'].shift(1) < data['SR_D'].shift(1)), 'SR_signal'] = 1  # Señal de compra
            data.loc[(data['SR_K'] < data['SR_D']) & (
                        data['SR_K'].shift(1) > data['SR_D'].shift(1)), 'SR_signal'] = -1  # Señal de venta
            signal_columns.append('SR_signal')

        if "DMI" in strats:
            window_dmi = params["DMI_window"]  # Ventana de tiempo para DMI
            adx_threshold = params["ADX_threshold"]  # Umbral para generar señales

            # Calcula los componentes del DMI: ADX, +DI y -DI
            dmi_indicator = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=window_dmi,
                                                  fillna=False)
            data['ADX'] = dmi_indicator.adx()
            data['DI+'] = dmi_indicator.adx_pos()
            data['DI-'] = dmi_indicator.adx_neg()

            # Define las señales de compra y venta basadas en el ADX y el cruce de +DI y -DI
            data['DMI_signal'] = 0
            data.loc[(data['DI+'] > data['DI-']) & (data['ADX'] > adx_threshold), 'DMI_signal'] = 1  # Señal de compra
            data.loc[(data['DI+'] < data['DI-']) & (data['ADX'] > adx_threshold), 'DMI_signal'] = -1  # Señal de venta
            signal_columns.append('DMI_signal')

        data['general_signals'] = data[signal_columns].mean(axis=1, skipna=True)

        data.loc[data['general_signals'] > 0.4, 'general_signals'] = 1
        data.loc[(data['general_signals'] >= -0.4) & (data['general_signals'] <= 0.4), 'general_signals'] = 0
        data.loc[data['general_signals'] < -0.4, 'general_signals'] = -1

        return data

    data = signal_functions(strats=strats, params=params, data=data)

    return data
