'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/16
'''


'''
绘制sigmoid函数
'''

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    # 直接返回sigmoid函数
    return 1 / (1 + np.exp(-x))


def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-8, 8, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    plot_sigmoid()
