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
        self.data = train_data
        self.eval_data = eval_data
        self.init_balance = init_balance

        self.len_obs = len_obs
        self.len_window = len_window
        self.action_space = spaces.Discrete(5)  # (-20%, -10%, 0, 10%, 20%), (% amount to buy if positive)
        self.observation_space = spaces.Box(-5000, 5000, shape=(self.len_obs, 1), dtype=np.float32)

        self.list_nw = []

    def reset(self, train_data=True, loc=0):
        # Reset the state of the environment to an initial state
        self.balance = self.init_balance
        self.net_worth = self.init_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.list_nw = np.array([])

        # Set the current step to a random point within the dataframe
        if train_data:
            self.current_step = random.randint(self.len_obs, len(self.data) - self.len_window)
        else:
            self.current_step = loc
        self.posit_window = 0
        return self._next_observation(train_data)

    def _next_observation(self, train_data=True):
        # Get data points for the last 20 days
        if train_data:
            frame = self.data.iloc[self.current_step - self.len_obs:self.current_step, [1]].values
        else:
            frame = self.eval_data.iloc[self.current_step - self.len_obs:self.current_step, [1]].values
        return frame.reshape(1, -1)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.posit_window += 1
        self.current_step += 1

        if self.posit_window == 1:
            reward = 0
        else:
            reward = (self.list_nw[-1] / self.list_nw[-2])-1
        done = (self.posit_window == self.len_window)
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        current_price = self.data.iloc[self.current_step, :].loc['prices']

        if 4 < action <= 5:
            # Buy 40% of balance in shares
            total_possible = self.balance / current_price  # nro acciones que podrías comprar con balance
            shares_bought = math.floor(total_possible * 0.40)  # 40% de lo anterior
            prev_cost = self.cost_basis * self.shares_held  # coste X cantidad de acciones que tengo
            additional_cost = shares_bought * current_price  # nro acciones a comprar X precio actual

            self.balance -= additional_cost  # restamos el gasto en acciones del balance
            self.cost_basis = (prev_cost + additional_cost)/(self.shares_held+shares_bought)  # coste medio de las acciones
            self.shares_held += shares_bought  # acciones totales que tengo

        elif 3 < action <= 4:
            # Buy 10% of balance in shares
            total_possible = self.balance / current_price
            shares_bought = math.floor(total_possible * 0.10)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost)/(self.shares_held+shares_bought)
            self.shares_held += shares_bought

        elif 2 < action <= 3:
            pass

        elif 1 < action <= 2:
            # Sell 10 % of shared held
            shares_sold = math.floor(self.shares_held * 0.10)  # tomamos 10% de las acciones que tenemos
            self.balance += shares_sold * current_price  # añadimos el dinero de la venta al balance
            self.shares_held -= shares_sold  # resto el número de acciones
            self.total_shares_sold += shares_sold  # sumas al número total de acciones vendidas
            self.total_sales_value += shares_sold * current_price  # valor de las acciones vendidas

        elif 0 < action <= 1:
            # Sell 40 % of shared held
            shares_sold = math.floor(self.shares_held * 0.40)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + (self.shares_held * current_price)
        self.list_nw = np.concatenate([self.list_nw, np.array([self.net_worth])])

        if self.shares_held == 0:
            self.cost_basis = 0

    def result(self, days_per_year=252):
        div = self.list_nw[-1] / self.list_nw[0]
        exp = (days_per_year / self.len_window)
        cagr = (div**exp) - 1
        ann_vol = ((self.list_nw[1:] / self.list_nw[:-1]) - 1).std() * (days_per_year**0.5)
        return cagr, ann_vol

    def render(self, ep, close=False):
        self.profit = (self.net_worth - self.init_balance) / self.init_balance

        # Render the environment to the screen
        print(f'\nep {ep} ' + '*' * 21)
        print(f'Balance: {round(self.balance, 2)}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {round(self.cost_basis, 2)} (Total sales value: {round(self.total_sales_value, 2)})')
        print(f'Net worth: {round(self.net_worth, 2)} (Max net worth: {round(max(self.list_nw), 2)})')
        print(f'Profit: {round(self.profit * 100, 2)}%')


