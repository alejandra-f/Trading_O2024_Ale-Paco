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
