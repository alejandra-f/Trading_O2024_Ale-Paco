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

# Variable global para almacenar el mejor resultado
global best_trial_info
best_trial_info = {'best_value': -float('inf'), 'best_combination': None, 'best_params': None}

# Función para cambiar el formato de fechas en múltiples datasets
def update_datetime_format(datasets: list):
    for data in datasets:
        try:
            data.rename(columns={'Date': 'Datetime'}, inplace=True)
        except:
            try:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
            except:
                continue

# Función para cambiar el formato de fechas en un dataset
def update_single_datetime_format(data):
    try:
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
    except:
        try:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
        except:
            print('Error al cambiar el formato de fecha')
    data = data.copy()
    return data

# Función principal de backtesting para un solo conjunto de parámetros
def perform_backtesting(params, data):
    # Copiar los datos para no modificar el original
    data = data.copy()
    update_single_datetime_format(data)

    # Obtener los hiperparámetros a optimizar
    take_profit_multiplier = params['take_profit_multiplier']
    stop_loss_multiplier = params['stop_loss_multiplier']
    n_shares_long = params['n_shares_long']
    n_shares_short = params['n_shares_short']

    # Inicializar variables del backtesting
    initial_cash = 1_000_000
    cash_balance = initial_cash
    margin_account = 0  # Dinero reservado para posiciones en corto
    active_positions = []
    trade_history = []
    portfolio_values = []
    commission_rate = 0.125 / 100
    margin_requirement = 0.25  # Porcentaje de margen requerido para posiciones cortas

    for i, row in data.iterrows():
        # Cálculo del margen requerido para las posiciones cortas
        total_margin_required = sum(position["n_shares"] * row.Close * margin_requirement 
                                    for position in active_positions if position["type"] == "short")

        # Verificar si se requiere transferir efectivo al margen
        if total_margin_required > margin_account:
            additional_margin_needed = total_margin_required - margin_account

            if cash_balance >= additional_margin_needed:
                # Transferir efectivo a la cuenta de margen
                cash_balance -= additional_margin_needed
                margin_account += additional_margin_needed
            else:
                # Si no hay suficiente efectivo, cerrar posiciones cortas hasta cubrir el margen
                for position in active_positions.copy():
                    if position["type"] == "short":
                        profit = (position["sold_at"] - row.Close) * position["n_shares"]
                        cash_balance += profit - (profit * commission_rate)  # Ajustar por comisión
                        margin_account -= row.Close * position["n_shares"] * margin_requirement  # Liberar margen
                        cash_balance += position["n_shares"] * row.Close * margin_requirement
                        trade_history.append({"Datetime": row.Datetime, "operation": "closed short", 
                                              "price": row.Close, "n_shares": position["n_shares"]})
                        active_positions.remove(position)
                        if sum(position["n_shares"] * row.Close * margin_requirement 
                               for position in active_positions if position["type"] == "short") <= (margin_account + cash_balance):
                            break  # Salir si el margen es suficiente

        # Revertir margen adicional si se ha liberado
        if total_margin_required < margin_account:
            excess_margin = margin_account - total_margin_required
            cash_balance += excess_margin
            margin_account -= excess_margin

        # Cerrar operaciones activas si alcanzan su stop loss o take profit
        for position in active_positions.copy():
            close_position = False
            if position["type"] == "long":
                if row.Close <= position["stop_loss"] or row.Close >= position["take_profit"]:
                    cash_balance += row.Close * position["n_shares"] * (1 - commission_rate)
                    close_position = True
            elif position["type"] == "short":
                if row.Close >= position["stop_loss"] or row.Close <= position["take_profit"]:
                    cash_balance += (position["sold_at"] - row.Close) * position["n_shares"] * (1 - commission_rate)
                    margin_account -= position["n_shares"] * row.Close * margin_requirement
                    cash_balance += position["n_shares"] * row.Close * margin_requirement
                    close_position = True

            if close_position:
                trade_history.append({"Datetime": row.Datetime, "operation": f"closed {position['type']}", 
                                      "price": row.Close, "n_shares": position["n_shares"]})
                active_positions.remove(position)

        # Abrir nuevas posiciones basadas en señales
        # Abrir posición larga
        if cash_balance > row.Close * n_shares_long * (1 + commission_rate):
            if row.general_signals == 1:
                active_positions.append({
                    "Datetime": row.Datetime,
                    "bought_at": row.Close,
                    "type": "long",
                    "n_shares": n_shares_long,
                    "stop_loss": row.Close * stop_loss_multiplier,
                    "take_profit": row.Close * take_profit_multiplier
                })
                cash_balance -= row.Close * n_shares_long * (1 + commission_rate)
                trade_history.append({"Datetime": row.Datetime, "operation": "opened long", 
                                      "price": row.Close, "n_shares": n_shares_long})

        # Abrir posición corta
        margin_for_short = row.Close * n_shares_short * margin_requirement
        if cash_balance >= margin_for_short:  
            if row.general_signals == -1:
                active_positions.append({
                    "Datetime": row.Datetime,
                    "sold_at": row.Close,
                    "type": "short",
                    "n_shares": n_shares_short,
                    "stop_loss": row.Close * stop_loss_multiplier,
                    "take_profit": row.Close * take_profit_multiplier,
                })
                margin_account += margin_for_short
                cash_balance -= margin_for_short
                trade_history.append({"Datetime": row.Datetime, "operation": "opened short", 
                                      "price": row.Close, "n_shares": n_shares_short})

        # Actualizar el valor de la cartera
        total_assets_value = sum([position["n_shares"] * row.Close for position in active_positions if position["type"] == "long"])
        portfolio_values.append(cash_balance + total_assets_value + margin_account)

    final_portfolio_value = portfolio_values[-1]
    return final_portfolio_value

# Función de optimización principal
def optimize_strategy(trial, data):
    global best_trial_info

    data = data.copy()
    combinations = list(itertools.product([0, 1], repeat=5))
    combinations = [''.join(map(str, comb)) for comb in combinations]
    combinations = combinations[1:]  # Omitir la primera combinación (00000)

    for combination in combinations:
        # Hiperparámetros a optimizar
        take_profit_multiplier = trial.suggest_float("take_profit_multiplier", 1.01, 1.1)
        stop_loss_multiplier = trial.suggest_float("stop_loss_multiplier", 0.9, 0.99)
        n_shares_long = trial.suggest_int("n_shares_long", 1, 100)
        n_shares_short = trial.suggest_int("n_shares_short", 1, 100)

        # Crear diccionario de parámetros
        strategy_params = {
            "combination": combination,
            "take_profit_multiplier": take_profit_multiplier,
            "stop_loss_multiplier": stop_loss_multiplier,
            "n_shares_long": n_shares_long,
            "n_shares_short": n_shares_short
        }
        
        selected_strategies = []

        # Definir las estrategias seleccionadas en base a la combinación
        if combination[0] == '1':
            selected_strategies.append("RSI")
            strategy_params["RSI"] = {
                "window": trial.suggest_int("RSI_window", 1, 30),
                "up_threshold": trial.suggest_int("RSI_up_threshold", 50, 80),
                "lw_threshold": trial.suggest_int("RSI_lw_threshold", 30, 50)
            }
        if combination[1] == '1':
            selected_strategies.append("BB")  # Bollinger Bands
            strategy_params["BB"] = {
                "window": trial.suggest_int("BB_window", 1, 30),
                "window_dev": trial.suggest_float("BB_window_dev", 1.0, 3.0)
            }
        if combination[2] == '1':
            selected_strategies.append("WMA")  # Weighted Moving Average
            strategy_params["WMA"] = {
                'window': trial.suggest_int("WMA_window", 1, 30)
            }
        if combination[3] == '1':
            selected_strategies.append("STO")  # Stochastic Oscillator
            strategy_params["STO"] = {
                "k_window": trial.suggest_int("k_window", 1, 30),
                "d_window": trial.suggest_int("d_window", 1, 20)
            }
        if combination[4] == '1':
            selected_strategies.append("DMI")  # Direccional Movement Index
            strategy_params["DMI"] = {
                "window": trial.suggest_int("DMI_window", 1, 30)
            }

        # Función para generar señales en función de las estrategias seleccionadas
        def generate_signals(selected_strategies=selected_strategies, strategy_params=strategy_params, data=data):
            signal_columns = []

            if "RSI" in selected_strategies:
                rsi_params = strategy_params["RSI"]
                window_rsi = rsi_params["window"]
                rsi_indicator = ta.momentum.RSIIndicator(data['Close'], window=window_rsi, fillna=False)
                data['RSI'] = rsi_indicator.rsi()

                data['RSI_signal'] = 0
                data.loc[data['Close'] > data['RSI'], 'RSI_signal'] = 1  # Señal de compra
                data.loc[data['Close'] < data['RSI'], 'RSI_signal'] = -1  # Señal de venta
                signal_columns.append('RSI_signal')

            if "BB" in selected_strategies:
                bb_params = strategy_params["BB"]
                window_bb = bb_params["window"]
                window_dev_bb = bb_params["window_dev"]

                bb_indicator = ta.volatility.BollingerBands(data['Close'], window=window_bb, window_dev=window_dev_bb, fillna=False)
                data['BB_hband'] = bb_indicator.bollinger_hband()
                data['BB_lband'] = bb_indicator.bollinger_lband()

                data['BB_signal'] = 0
                data.loc[data['Close'] < data['BB_lband'], 'BB_signal'] = 1  # Señal de compra
                data.loc[data['Close'] > data['BB_hband'], 'BB_signal'] = -1  # Señal de venta
                signal_columns.append('BB_signal')

            if "WMA" in selected_strategies:
                wma_params = strategy_params["WMA"]
                window_wma = wma_params["window"]
                wma_indicator = ta.trend.WMAIndicator(data['Close'], window=window_wma, fillna=False)
                data['WMA'] = wma_indicator.wma()

                data['WMA_signal'] = 0
                data.loc[data['Close'] > data['WMA'], 'WMA_signal'] = 1  # Señal de compra
                data.loc[data['Close'] < data['WMA'], 'WMA_signal'] = -1  # Señal de venta
                signal_columns.append('WMA_signal')

            if "STO" in selected_strategies:
                sto_params = strategy_params["STO"]
                k_window_sto = sto_params["k_window"]
                d_window_sto = sto_params["d_window"]

                sr_indicator = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=k_window_sto, smooth_window=d_window_sto, fillna=False)
                data['SR_K'] = sr_indicator.stoch()
                data['SR_D'] = sr_indicator.stoch_signal()

                data['SR_signal'] = 0
                data.loc[(data['SR_K'] > data['SR_D']) & (data['SR_K'].shift(1) < data['SR_D'].shift(1)), 'SR_signal'] = 1  # Señal de compra
                data.loc[(data['SR_K'] < data['SR_D']) & (data['SR_K'].shift(1) > data['SR_D'].shift(1)), 'SR_signal'] = -1  # Señal de venta
                signal_columns.append('SR_signal')

            if "DMI" in selected_strategies:
                dmi_params = strategy_params["DMI"]
                window_dmi = dmi_params["window"]

                adx_indicator = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=window_dmi, fillna=False)
                data['ADX'] = adx_indicator.adx()
                data['DI+'] = adx_indicator.adx_pos()
                data['DI-'] = adx_indicator.adx_neg()

                data['DMI_signal'] = 0
                data.loc[(data['DI+'] > data['DI-']) & (data['ADX'] > 20), 'DMI_signal'] = 1  # Señal de compra
                data.loc[(data['DI+'] < data['DI-']) & (data['ADX'] > 20), 'DMI_signal'] = -1  # Señal de venta
                signal_columns.append('DMI_signal')

            data['general_signals'] = data[signal_columns].mean(axis=1, skipna=True)
            data.loc[data['general_signals'] > 0.4, 'general_signals'] = 1
            data.loc[(data['general_signals'] >= -0.4) & (data['general_signals'] <= 0.4), 'general_signals'] = 0
            data.loc[data['general_signals'] < -0.4, 'general_signals'] = -1

            return data

        # Generar señales y ejecutar backtesting
        data = generate_signals(selected_strategies=selected_strategies, strategy_params=strategy_params, data=data)
        portfolio_value = perform_backtesting(strategy_params, data)

        # Almacenar la mejor combinación
        if portfolio_value > best_trial_info['best_value']:
            best_trial_info['best_value'] = portfolio_value
            best_trial_info['best_combination'] = combination
            best_trial_info['best_params'] = strategy_params.copy()

            # Guardar información adicional en user_attrs
            trial.set_user_attr('best_combination', combination)

    return best_trial_info['best_value']

def final_backtesting(strategy_params, data):
    # Copiar los datos para evitar modificar el original
    data = data.copy()

    # Definir hiperparámetros a optimizar
    take_profit_multiplier = strategy_params['take_profit_multiplier']
    stop_loss_multiplier = strategy_params['stop_loss_multiplier']
    n_shares_long = strategy_params['n_shares_long']
    n_shares_short = strategy_params['n_shares_short']

    # Inicializar variables para el backtesting
    cash_balance = 10_000  # Capital inicial
    margin_account = 0  # Cuenta de margen para operaciones en corto
    active_positions = []  # Lista de operaciones activas
    trade_history = []  # Registro de operaciones realizadas
    portfolio_values = []  # Valores de la cartera a lo largo del tiempo
    commission_rate = 0.125 / 100  # Comisión
    margin_requirement = 0.25  # Porcentaje de margen requerido

    # Iterar sobre cada fila de datos
    for i, row in data.iterrows():
        # Actualizar el margen necesario para las posiciones cortas
        total_margin_required = sum(position["n_shares"] * row.Close * margin_requirement 
                                    for position in active_positions if position["type"] == "short")

        # Verificar si es necesario transferir efectivo al margen
        if total_margin_required > margin_account:
            additional_margin_needed = total_margin_required - margin_account
            if cash_balance >= additional_margin_needed:
                cash_balance -= additional_margin_needed
                margin_account += additional_margin_needed
            else:
                # Cerrar posiciones cortas hasta cubrir el margen
                for position in active_positions.copy():
                    if position["type"] == "short":
                        profit = (position["sold_at"] - row.Close) * position["n_shares"]
                        cash_balance += profit - (profit * commission_rate)
                        margin_account -= row.Close * position["n_shares"] * margin_requirement
                        cash_balance += position["n_shares"] * row.Close * margin_requirement
                        trade_history.append({"Datetime": row.Datetime, "operation": "closed short", 
                                              "price": row.Close, "n_shares": position["n_shares"]})
                        active_positions.remove(position)
                        if sum(position["n_shares"] * row.Close * margin_requirement 
                               for position in active_positions if position["type"] == "short") <= (margin_account + cash_balance):
                            break  # Salir si el margen es suficiente

        # Revertir margen si sobra
        if total_margin_required < margin_account:
            excess_margin = margin_account - total_margin_required
            cash_balance += excess_margin
            margin_account -= excess_margin

        # Cerrar operaciones activas si alcanzan su stop loss o take profit
        for position in active_positions.copy():
            close_position = False
            if position["type"] == "long":
                if row.Close <= position["stop_loss"] or row.Close >= position["take_profit"]:
                    cash_balance += row.Close * position["n_shares"] * (1 - commission_rate)
                    close_position = True
            elif position["type"] == "short":
                if row.Close >= position["stop_loss"] or row.Close <= position["take_profit"]:
                    cash_balance += (position["sold_at"] - row.Close) * position["n_shares"] * (1 - commission_rate)
                    margin_account -= position["n_shares"] * row.Close * margin_requirement
                    cash_balance += position["n_shares"] * row.Close * margin_requirement
                    close_position = True

            if close_position:
                trade_history.append({"Datetime": row.Datetime, "operation": f"closed {position['type']}", 
                                      "price": row.Close, "n_shares": position["n_shares"]})
                active_positions.remove(position)

        # Abrir nuevas operaciones según las señales
        # Abrir posición larga
        if cash_balance > row.Close * n_shares_long * (1 + commission_rate):
            if row.general_signals == 1:
                active_positions.append({
                    "Datetime": row.Datetime,
                    "bought_at": row.Close,
                    "type": "long",
                    "n_shares": n_shares_long,
                    "stop_loss": row.Close * stop_loss_multiplier,
                    "take_profit": row.Close * take_profit_multiplier
                })
                cash_balance -= row.Close * n_shares_long * (1 + commission_rate)
                trade_history.append({"Datetime": row.Datetime, "operation": "opened long", 
                                      "price": row.Close, "n_shares": n_shares_long})

        # Abrir posición corta
        margin_for_short = row.Close * n_shares_short * margin_requirement
        if cash_balance >= margin_for_short:
            if row.general_signals == -1:
                active_positions.append({
                    "Datetime": row.Datetime,
                    "sold_at": row.Close,
                    "type": "short",
                    "n_shares": n_shares_short,
                    "stop_loss": row.Close * stop_loss_multiplier,
                    "take_profit": row.Close * take_profit_multiplier
                })
                margin_account += margin_for_short
                cash_balance -= margin_for_short
                trade_history.append({"Datetime": row.Datetime, "operation": "opened short", 
                                      "price": row.Close, "n_shares": n_shares_short})

        # Actualizar el valor de la cartera
        total_assets_value = sum([position["n_shares"] * row.Close for position in active_positions if position["type"] == "long"])
        portfolio_values.append(cash_balance + total_assets_value + margin_account)

    final_portfolio_value = portfolio_values[-1]
    return final_portfolio_value, portfolio_values

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

    def generate_signals(selected_strategies=selected_strategies, strategy_params=strategy_params, data=data):
        
        signal_columns = []
        
        if "RSI" in selected_strategies:
            window_rsi = strategy_params["RSI_window"]
            up_threshold_rsi = strategy_params["RSI_up_threshold"]
            lw_threshold_rsi = strategy_params["RSI_lw_threshold"]
            
            rsi_indicator = ta.momentum.RSIIndicator(data['Close'], window=window_rsi, fillna=False)
            data['RSI'] = rsi_indicator.rsi()

            # Definición de señales
            data['RSI_signal'] = 0
            data.loc[data['RSI'] < up_threshold_rsi, 'RSI_signal'] = 1  # Señal de compra
            data.loc[data['RSI'] > lw_threshold_rsi, 'RSI_signal'] = -1  # Señal de venta
            signal_columns.append('RSI_signal')

            
        if "BB" in selected_strategies:
            window_bb = strategy_params["BB_window"]
            window_bb_dev = strategy_params["BB_window_dev"]

            bb_indicator = ta.volatility.BollingerBands(data['Close'], window=window_bb, window_dev=window_bb_dev, fillna=False)
            data['BB_hband'] = bb_indicator.bollinger_hband()
            data['BB_lband'] = bb_indicator.bollinger_lband()

            # Definir señales basadas en las Bandas de Bollinger
            data['BB_signal'] = 0
            data.loc[data['Close'] < data['BB_lband'], 'BB_signal'] = 1  # Señal de compra
            data.loc[data['Close'] > data['BB_hband'], 'BB_signal'] = -1  # Señal de venta
            signal_columns.append('BB_signal')

            
        if "WMA" in selected_strategies:
            window_wma = strategy_params["WMA_window"]
            
            wma_indicator = ta.trend.WMAIndicator(data['Close'], window=window_wma, fillna=False)
            data['WMA'] = wma_indicator.wma()

            # Definir señales basadas en el WMA
            data['WMA_signal'] = 0
            data.loc[data['Close'] > data['WMA'], 'WMA_signal'] = 1  # Señal de compra
            data.loc[data['Close'] < data['WMA'], 'WMA_signal'] = -1  # Señal de venta
            signal_columns.append('WMA_signal')

            
        if "STO" in selected_strategies:
            k_window_sto = strategy_params["k_window"]
            d_window_sto = strategy_params["d_window"]

            sto_indicator = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=k_window_sto, smooth_window=d_window_sto, fillna=False)
            data['STO_K'] = sto_indicator.stoch()
            data['STO_D'] = sto_indicator.stoch_signal()

            # Definir señales basadas en el Stochastic Oscillator
            data['STO_signal'] = 0
            data.loc[(data['STO_K'] > data['STO_D']) & (data['STO_K'].shift(1) < data['STO_D'].shift(1)), 'STO_signal'] = 1  # Señal de compra
            data.loc[(data['STO_K'] < data['STO_D']) & (data['STO_K'].shift(1) > data['STO_D'].shift(1)), 'STO_signal'] = -1  # Señal de venta
            signal_columns.append('STO_signal')
            
        if "DMI" in selected_strategies:
            window_dmi = strategy_params["DMI_window"]
            adx_threshold = strategy_params["ADX_threshold"]
            
            # Calcular componentes del DMI
            dmi_indicator = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=window_dmi, fillna=False)
            data['ADX'] = dmi_indicator.adx()
            data['DI+'] = dmi_indicator.adx_pos()
            data['DI-'] = dmi_indicator.adx_neg()
            
            # Definir señales basadas en el ADX y DI+/DI-
            data['DMI_signal'] = 0
            data.loc[(data['DI+'] > data['DI-']) & (data['ADX'] > adx_threshold), 'DMI_signal'] = 1  # Señal de compra
            data.loc[(data['DI+'] < data['DI-']) & (data['ADX'] > adx_threshold), 'DMI_signal'] = -1  # Señal de venta
            signal_columns.append('DMI_signal')

        # Combinación de señales de todas las estrategias
        data['general_signals'] = data[signal_columns].mean(axis=1, skipna=True)

        # Definir señales generales basadas en umbrales
        data.loc[data['general_signals'] > 0.4, 'general_signals'] = 1
        data.loc[(data['general_signals'] >= -0.4) & (data['general_signals'] <= 0.4), 'general_signals'] = 0
        data.loc[data['general_signals'] < -0.4, 'general_signals'] = -1

        return data

    # Generar señales con las estrategias seleccionadas
    data = generate_signals(selected_strategies=selected_strategies, strategy_params=strategy_params, data=data)

    return data
