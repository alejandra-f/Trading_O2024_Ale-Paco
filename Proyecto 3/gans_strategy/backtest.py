import pandas as pd

class Backtest:
    def __init__(self, data, initial_balance=10000):
        """
        Inicializa el backtest con los datos de mercado y un saldo inicial.
        """
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # Posición actual (1 para largo, -1 para corto)
        self.history = []  # Historial de operaciones

    def run(self):
        """
        Ejecuta el backtest de la estrategia en los datos dados.
        """
        for i in range(1, len(self.data)):
            signal = self.data['Signal'].iloc[i]
            price = self.data['Close'].iloc[i]

            if signal == 1 and self.position == 0:  # Señal de compra
                self.position = self.balance / price
                self.balance -= self.position * price
                self.history.append(('Compra', price, self.balance))

            elif signal == -1 and self.position > 0:  # Señal de venta
                self.balance += self.position * price
                self.history.append(('Venta', price, self.balance))
                self.position = 0

        # Cerrar cualquier posición restante al final
        if self.position > 0:
            self.balance += self.position * self.data['Close'].iloc[-1]
            self.position = 0

        return self.balance

    def get_history(self):
        """
        Devuelve el historial de operaciones como un DataFrame de pandas.
        """
        return pd.DataFrame(self.history, columns=['Acción', 'Precio', 'Balance'])