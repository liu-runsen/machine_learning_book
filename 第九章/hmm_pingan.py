'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/9
'''
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
from hmmlearn.hmm import GaussianHMM
# 平安股票代码601318
from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator

data = ts.get_k_data('601318',start = '2015-01-01',end='2019-12-31')
# dates = time.mktime(data['date'].values)
#当日收盘价格
close_v = data['close'].values
#当日交易量
volume = data['volume'].values
# print(dates)

diff = np.diff(close_v)
# print(diff.shape,volume.shape)
X = np.column_stack([diff, volume[1:]])
# print(X)

# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib as mpl
# from sklearn.metrics.pairwise import pairwise_distances_argmin
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore")   # hmmlearn(0.2.0) < sklearn(0.18)

    # 0日期  1开盘  2最高  3最低  4收盘  5成交量  6成交额
    x = np.loadtxt('SH600000.txt', delimiter='\t', skiprows=2, usecols=(4, 5, 6, 2, 3))
    close_price = x[:, 0]
    volumn = x[:, 1]
    amount = x[:, 2]
    amplitude_price = x[:, 3] - x[:, 4] # 每天的最高价与最低价的差
    diff_price = np.diff(close_price)   # 涨跌值
    volumn = volumn[1:]                 # 成交量
    amount = amount[1:]                 # 成交额
    amplitude_price = amplitude_price[1:]   # 每日振幅
    sample = np.column_stack((diff_price, volumn, amount, amplitude_price))    # 观测值
    n = 5
    model = hmm.GaussianHMM(n_components=n, covariance_type='full')
    model.fit(sample)
    y = model.predict_proba(sample)
    np.set_printoptions(suppress=True)
    print(y)

    t = np.arange(len(diff_price))
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,7), facecolor='w')
    plt.subplot(421)
    plt.plot(t, diff_price, 'r-', lw=0.7)
    plt.grid(True)
    plt.title('涨跌幅')
    plt.subplot(422)
    plt.plot(t, volumn, 'g-', lw=0.7)
    plt.grid(True)
    plt.title('交易量')

    clrs = plt.cm.terrain(np.linspace(0, 0.8, n))
    plt.subplot(423)
    for i, clr in enumerate(clrs):
        plt.plot(t, y[:, i], '-', color=clr, alpha=0.7, lw=0.7)
    plt.title('所有组分')
    plt.grid(True)
    for i, clr in enumerate(clrs):
        axes = plt.subplot(4, 2, i+4)
        plt.plot(t, y[:, i], '-', color=clr, lw=0.7)
        plt.title('组分%d' % (i+1))
        plt.grid(True)
    # plt.suptitle('SH600000股票：GaussianHMM分解隐变量', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
