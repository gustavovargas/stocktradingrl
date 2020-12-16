import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import os
import math
import random
import tensorflow as tf
import gym
import argparse
import stock_env

sns.set()  # just for better images


def get_data(path, index_col=0, train_pct=0.7):
    """

    :param train_pct:
    :param index_col:
    :param path:
    :return:
    """
    data = pd.read_csv(path, index_col=index_col, parse_dates=True, header=0)
    data = pd.concat([data, data.pct_change()], axis=1).iloc[1:]
    data.columns = ['prices', 'returns']
    sep = math.floor(len(data) * train_pct)
    train_data = data.iloc[:sep, :]
    test_data = data.iloc[sep:, :]
    return train_data.values, test_data.values


def get_performance(env, agent, train_data=True, training=False, batch_size=12, overlap=20):
    state = env.reset(train_data=train_data, batch_size=batch_size, overlap=overlap)
    done = False
    while not done:
        action = agent.get_action(state, use_random=True)
        next_state, reward, done, info = env.step(action)
        if training:
            agent.train((state, action, next_state, reward, done))
        state = next_state
    cagr, vol = env.result()
    return cagr, vol


def check_directories(name_project: str, len_obs:str, len_window:str):
    name_project = f'{name_project}_obs{len_obs}_window{len_window}'
    directories = [f'projects/{name_project}/{dir}/' for dir in ['saved_models', 'imgs', 'statistics']]
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)
    return directories


class QNetwork(tf.keras.Model):
    def __init__(self, state_shape, action_size, learning_rate):
        super().__init__()
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.q_first = tf.keras.layers.Dense(units=100, input_shape=self.state_shape)
        self.q_second = tf.keras.layers.Dense(units=100, input_shape=(100,))
        self.q_state = tf.keras.layers.Dense(units=self.action_size, name='q_table', input_shape=(100,))

    def __call__(self, inputs):
        self.state_in, self.action_in, self.target_in = inputs
        self.action = tf.one_hot(self.action_in, depth=self.action_size)
        self.q_first_layer = self.q_first(self.state_in)
        self.q_second_layer = self.q_second(self.q_first_layer)
        self.q_state_layer = self.q_state(self.q_second_layer)

        self.q_action = tf.reduce_sum(input_tensor=tf.multiply(self.q_state_layer, self.action), axis=1)

        return self.q_action


class QNAgent:
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        self.action_size = env.action_space.n
        print('Action_size: ', self.action_size)
        self.state_size = env.observation_space.shape
        print('State size: ', self.state_size)

        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.model = QNetwork(self.state_size, self.action_size, self.learning_rate)

    def get_action(self, state, use_random=True):
        """Select action based on the q value corresponding to a given state. Best
        action will be the index of the highest q_value. Use np.argmax to take that."""

        q_state = self.model.q_state(self.model.q_second(self.model.q_first(state)))
        action_greedy = np.argmax(q_state, axis=1)
        if use_random:
            action_random = [random.choice(range(self.action_size)) for i in range(len(state))]
            return action_random if random.random() < 0.90 else action_greedy
        else:
            return action_greedy

    def train(self, experience: tuple):
        state, action, next_state, reward, done = (exp for exp in experience)
        q_next = self.model.q_state(self.model.q_second(self.model.q_first(state)))
        q_target = reward + self.discount_rate * np.max(q_next, axis=1)

        with tf.GradientTape() as tape:
            q_action = self.model([state, action, q_target])
            loss = tf.reduce_sum(input_tensor=tf.square(q_target - q_action))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def load(self, path: str):
        self.model.load_weights(path)


def plot_stocks_trading_performance(data, save_name, color='royalblue', alpha=0.5, s=12, acc_title=''):
    plt.scatter(data[:, 2] * 100, data[:, 1] * 100, alpha=alpha, s=s, color=color)
    plt.xlabel('Volatility')
    plt.ylabel('CAGR')
    plt.title(f'RL Trading - Episode {ep} {acc_title}')
    plt.xlim(0, 50)
    plt.ylim(-50, 100)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    plt.cla()
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Trading with Q Learning')
    parser.add_argument('-name_file_data', default='AXP.csv', type=str)
    parser.add_argument('-name_project', default='AXP_batch', type=str)
    parser.add_argument('-env_name', type=str, default='StockEnv-v0')
    parser.add_argument('-num_episodes', type=int, default=100000)
    parser.add_argument('-len_obs', type=int, default=50)
    parser.add_argument('-len_window', type=int, default=100)
    parser.add_argument('-interval', type=int, default=1000)
    parser.add_argument('-load_model', type=bool, default=False)
    parser.add_argument('-epoch_to_load', type=int, default=10000)
    parser.add_argument('-name_model_weights', type=str, default='trading_weights')
    parser.add_argument('-overlap', type=int, default=20)

    args = parser.parse_args()
    path_models, path_imgs, path_stats = check_directories(args.name_project, args.len_obs, args.len_window)

    # Read data you want to use for trading
    train_data, test_data = get_data(f'data/{args.name_file_data}')

    # Create an instant
    env = gym.make(args.env_name, train_data=train_data, eval_data=test_data, len_obs=args.len_obs, len_window=args.len_window)
    print(f'Observation space: {env.observation_space}')
    print(f'Action space: {env.action_space}')

    # Create an agent
    agent = QNAgent(env)

    # Load model
    if args.load_model:
        agent.load(path_models + args.name_model_weights + f'_{args.epoch_to_load}')
        train_statistics = pd.read_csv(path_stats + 'train.csv')
        test_statistics = pd.read_csv(path_stats + 'test.csv')
        init_ep = args.epoch_to_load
    else:
        init_ep = 0
        train_statistics = pd.DataFrame()
        test_statistics = pd.DataFrame()

    for ep in range(init_ep, args.num_episodes):
        get_performance(env, agent, train_data=True, training=True, batch_size=12)
        env.render(ep)

        if (ep % args.interval == 0) and not((args.load_model==True) and (ep == args.epoch_to_load)):
            agent.model.save_weights(path_models + args.name_model_weights + f'_{ep}')

            overlap = args.overlap
            results_train = np.empty(shape=(0, 3))
            results_test = np.empty(shape=(0, 3))

            size_test = ((len(env.eval_data)-env.len_obs-env.len_window) // overlap)+1
            cagr_train, vol_train = get_performance(env, agent, train_data=True, training=False, batch_size=size_test)
            results_train = np.array([np.tile(ep, size_test), cagr_train, vol_train]).transpose()

            cagr_test, vol_test = get_performance(env, agent, train_data=False, training=False, overlap=overlap, batch_size=size_test)
            results_test = np.array([np.tile(ep, size_test), cagr_test, vol_test]).transpose()

            train_statistics = pd.concat([train_statistics, pd.DataFrame(results_train, columns=['epoch', 'cagr','volatility'])])
            train_statistics.to_csv(path_stats+'train.csv', index=False)
            test_statistics = pd.concat([test_statistics, pd.DataFrame(results_test, columns=['epoch', 'cagr','volatility'])])
            test_statistics.to_csv(path_stats+'test.csv', index=False)

            plot_stocks_trading_performance(results_train, path_imgs + f'train_cagr_vol_ep_{ep}',
                                            color='royalblue', acc_title='Train')
            plot_stocks_trading_performance(results_test, path_imgs + f'test_cagr_vol_ep_{ep}',
                                            color='firebrick', acc_title='Test')

