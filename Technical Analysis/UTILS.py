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
best_trial_info={'best_value': -float(inf), 'best_combination': None, 'best_params':None}

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
    take_profit_multiplier=paras['take_profit_multiplier']
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

