'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
def calc_e_small(x):
    '''
    计算前10个
    '''
    n = 10
    f = np.arange(1, n + 1).cumprod()  # 阶乘
    b = np.array([x] * n).cumprod()  # 算的是x的n次方
    return np.sum(b / f) + 1


def calc_e(x):
    reverse = False
    if x < 0:  # 处理负数
        x = -x
        reverse = True
    y = calc_e_small(x)
    if reverse:
        return 1 / y
    return y


if __name__ == "__main__":
    t1 = np.linspace(-2, 0, 10, endpoint=False)
    t2 = np.linspace(0, 4, 20)
    t = np.concatenate((t1, t2))
    y = np.empty_like(t)
    for i, x in enumerate(t):
        y[i] = calc_e(x)
        print('e^', x, ' = ', y[i], '(近似值)\t', math.exp(x), '(真实值)')
    plt.figure(facecolor='w')
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y, 'r-', t, y, 'go', linewidth=2, markeredgecolor='k')
    plt.title('Taylor展式的应用 - $e^x$函数', fontsize=18)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('exp(X)', fontsize=15)
    plt.grid(True, ls=':')
    plt.show()
