'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''

from scipy import stats
import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

r = 1 / 50000
X = []
Y = []
for x in np.linspace(0, 1000000, 100000):
    if x == 0:
        continue
    # 直接用公式算
    # p = r*math.e**(-r*x)
    p = stats.expon.pdf(x, scale=1 / r)
    X.append(x)
    Y.append(p)
plt.plot(X, Y)
plt.xlabel("间隔时间")
plt.ylabel("概率密度")
plt.show()


for prob in range(3, 10, 3):
   x = np.arange(0, 25)
   binom = stats.binom.pmf(x, 20, 0.1*prob)
   plt.plot(x, binom, '-o', label="p = {:f}".format(0.1*prob))
   plt.xlabel('Random Variable', fontsize=12)
   plt.ylabel('Probability', fontsize=12)
   plt.title("Binomial Distribution varying p")
   plt.legend()
plt.show()


n = np.arange(-50, 50)
mean = 0
normal = stats.norm.pdf(n, mean, 10)
plt.plot(n, normal)
plt.xlabel('Distribution', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title("Normal Distribution")
plt.show()
