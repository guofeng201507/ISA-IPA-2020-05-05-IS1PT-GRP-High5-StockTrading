import os
import pickle
import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN
from rlenv.StockTradingEnv0 import StockTradingEnv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False

INITIAL_ACCOUNT_BALANCE = 10000


def stock_trade(stock_file_train):
    df_train = pd.read_csv(stock_file_train)
    df_train = df_train.sort_values('date')

    # The algorithms require a vectorized environment to run
    env_train = DummyVecEnv([lambda: StockTradingEnv(df_train)])

    model = PPO2(MlpPolicy, env_train, verbose=0, tensorboard_log='./log')
    # model = DQN("MlpPolicy", env_train, verbose=0, tensorboard_log='./log')
    model.learn(total_timesteps=int(1e4))

    # -----------------Test Model --------------------------------------
    day_profits = []
    buy_hold_profit = []

    df_test = pd.read_csv(stock_file_train.replace('train', 'test'))

    env_test = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env_test.reset()
    no_of_shares = 0
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        profit = env_test.render()
        day_profits.append(profit)
        if i == 0:
            buy_hold_profit.append(0)
            no_of_shares = INITIAL_ACCOUNT_BALANCE // df_test.iloc[0]['close']
            print('Buy ' + str(no_of_shares) + ' shares and hold')
        else:
            buy_hold_profit.append(no_of_shares * (df_test.iloc[i]['close'] - df_test.iloc[i - 1]['close']))
        if done:
            break
    return day_profits, buy_hold_profit


def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_a_stock_trade(stock_code):
    stock_file_train = find_file('./stockdata/train', str(stock_code))

    daily_profits, buy_hold_profit = stock_trade(stock_file_train)
    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.plot(buy_hold_profit, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='blue')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./img/{stock_code}.png')


def multi_stock_trade():
    start_code = 600000
    max_num = 3000

    group_result = []

    for code in range(start_code, start_code + max_num):
        stock_file = find_file('./stockdata/train', str(code))
        if stock_file:
            try:
                profits = stock_trade(stock_file)
                group_result.append(profits)
            except Exception as err:
                print(err)

    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:
        pickle.dump(group_result, f)


if __name__ == '__main__':
    # multi_stock_trade()
    test_a_stock_trade('sh.600028')
    # ret = find_file('./stockdata/train', '600036')
    # print(ret)
