import gym
from gym import spaces
import numpy as np
import random
import math


class CustomEnv(gym.Env):
    """A stock trading environment based on the one made up by Adam King
    (github.com/notadamking/Stock-Trading-Environment) for OpenAI gym
    """

    def __init__(self, train_data, eval_data, len_obs=20, len_window=100, init_balance=1000):
        super(CustomEnv, self).__init__()
        self.train_data = train_data
        self.eval_data = eval_data
        self.init_balance = init_balance

        self.len_obs = len_obs
        self.len_window = len_window
        self.action_space = spaces.Discrete(5)  # (-20%, -10%, 0, 10%, 20%), (% amount to buy if positive)
        self.observation_space = spaces.Box(-5000, 5000, shape=(self.len_obs, 1), dtype=np.float32)

        self.list_nw = []

    def reset(self, train_data=True, batch_size=12, overlap=20):
        # Reset the state of the environment to an initial state
        self.batch_size= batch_size
        self.balance = [self.init_balance for i in range(batch_size)]
        self.net_worth = [self.init_balance for i in range(batch_size)]
        self.shares_held = [0 for i in range(batch_size)]
        self.cost_basis = [0 for i in range(batch_size)]
        self.total_shares_sold = [0 for i in range(batch_size)]
        self.total_sales_value = [0 for i in range(batch_size)]
        self.list_nw = np.array([])

        # Set the current step to a random point within the dataframe

        if train_data:
            idx = np.random.randint(self.len_obs, len(self.train_data)-self.len_window, (batch_size,))
            self.prices = np.array([self.train_data[i-self.len_obs:i+self.len_window, 0] for i in idx])
            self.returns = np.array([self.train_data[i-self.len_obs:i+self.len_window, 1] for i in idx])
        else:
            idx = np.arange(self.len_obs, len(self.eval_data)-self.len_window, overlap)
            self.prices = np.array([self.eval_data[i-self.len_obs:i+self.len_window, 0] for i in idx])
            self.returns = np.array([self.eval_data[i-self.len_obs:i+self.len_window, 1] for i in idx])
        self.posit_window = 0
        return self._next_observation()

    def _next_observation(self):
        # Get data points for the last 20 days
        frame = self.returns[:, self.posit_window:self.posit_window + self.len_obs]
        return frame

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.posit_window += 1

        if self.posit_window == 1:
            reward = [0 for i in range(self.batch_size)]
        else:
            reward = (self.list_nw[:, -1] / self.list_nw[:, -2])-1
        done = (self.posit_window == self.len_window)
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        current_price = self.prices[:, self.posit_window+self.len_obs]

        for idx, acc in enumerate(action):
            if 4 < acc <= 5:
                # Buy 40% of balance in shares
                total_possible = self.balance[idx] / current_price[idx]  # nro acciones que podrías comprar con balance
                shares_bought = math.floor(total_possible * 0.40)  # 40% de lo anterior
                prev_cost = self.cost_basis[idx] * self.shares_held[idx]  # coste X cantidad de acciones que tengo
                additional_cost = shares_bought * current_price[idx]  # nro acciones a comprar X precio actual

                self.balance[idx] -= additional_cost  # restamos el gasto en acciones del balance
                self.cost_basis[idx] = (prev_cost + additional_cost)/(self.shares_held[idx] + shares_bought)  # coste medio de las acciones
                self.shares_held[idx] += shares_bought  # acciones totales que tengo

            elif 3 < acc <= 4:
                # Buy 10% of balance in shares
                total_possible = self.balance[idx] / current_price[idx]
                shares_bought = math.floor(total_possible * 0.40)
                prev_cost = self.cost_basis[idx] * self.shares_held[idx]
                additional_cost = shares_bought * current_price[idx]

                self.balance[idx] -= additional_cost
                self.cost_basis[idx] = (prev_cost + additional_cost)/(self.shares_held[idx] + shares_bought)
                self.shares_held[idx] += shares_bought

            elif 2 < acc <= 3:
                pass

            elif 1 < acc <= 2:
                # Sell 10 % of shared held
                try:
                    shares_sold = math.floor(self.shares_held[idx] * 0.10)
                except:
                    print(self.shares_held)
                    print(idx)  # tomamos 10% de las acciones que tenemos
                self.balance[idx] += shares_sold * current_price[idx]  # añadimos el dinero de la venta al balance
                self.shares_held[idx] -= shares_sold  # resto el número de acciones
                self.total_shares_sold[idx] += shares_sold  # sumas al número total de acciones vendidas
                self.total_sales_value[idx] += shares_sold * current_price[idx]  # valor de las acciones vendidas

            elif 0 < acc <= 1:
                # Sell 40 % of shared held
                try:
                    shares_sold = math.floor(self.shares_held[idx] * 0.40)
                except:
                    print(self.shares_held)
                    print(idx)
                self.balance[idx] += shares_sold * current_price[idx]
                self.shares_held[idx] -= shares_sold
                self.total_shares_sold[idx] += shares_sold
                self.total_sales_value[idx] += shares_sold * current_price[idx]

        self.net_worth = self.balance + (self.shares_held * current_price)
        if self.posit_window == 0:
            self.list_nw = np.array(self.net_worth).reshape(-1, 1)
        else:
            self.list_nw = np.concatenate([self.list_nw, np.array(self.net_worth).reshape(-1, 1)], axis=1)

        if self.shares_held == [0 for i in range(len(self.shares_held))]:
            self.cost_basis = [0 for i in range(self.batch_size)]

    def result(self, days_per_year=252):
        div = self.list_nw[:, -1] / self.list_nw[:, 0]
        exp = (days_per_year / self.len_window)
        cagr = (div**exp) - 1
        ann_vol = ((self.list_nw[:, 1:] / self.list_nw[:, :-1]) - 1).std(axis=1) * (days_per_year**0.5)  # !!
        return cagr, ann_vol

    def render(self, ep, close=False):
        profit = (self.net_worth - self.init_balance) / self.init_balance
        profit = np.mean(profit)
        vol_profit = np.std(profit)

        # Render the environment to the screen
        print(f'\nep {ep} ' + '*' * 21)
        print(f'Profit: {round(profit * 100, 2)}%')
        print(f'Vol profit: {round(vol_profit * 100, 2)}%')


