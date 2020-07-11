'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''

import numpy as np
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt
x = np.linspace( -3, 3, 100)
plt.plot(x, t.pdf(x,1), label='t=1')
plt.plot(x, t.pdf(x,2), label='t=2')
plt.plot(x, t.pdf(x,5), label = 't=5')
plt.plot(x, t.pdf(x,10), label = 't=10')
plt.plot( x[::5], norm.pdf(x[::5]),'kx', label='normal')
plt.legend()
plt.show()


import numpy as np
from scipy import stats
data = [23.68,23.98,23.72,21.98,23.79,25.48,24.28,23.75,23.74,23.92,22.86,22.03,24.26,22.45,23.99]
print(np.min(data),np.max(data))
print(stats.ttest_1samp(data,[np.min(data),np.max(data)]))


# 区间估计，计算95%保证程度下的区间估计范围。
se = np.std(data) / len(data) ** 0.5
# 均值下限
LB = np.mean(data) - 1.96 * se
# 均值上限
UB = np.mean(data) + 1.96 * se
print(LB, UB)



from scipy.stats import ttest_ind
x = [84.6,87.1,93.0,89.8,90.4,80.0,86.4,91.2,84.6,87.8]
y = [77.1,83.7,74.4,80.4,89.4,72.8,82.2,90.5,80.4,82.1]
# 方差齐性检验：当两总体方差相等时，即具有“方差齐性”，可以直接检验
# 独立样本T检验,默认方差齐性
print(ttest_ind(x, y))
# 如果方差不齐性，则equal_var=False
print(ttest_ind(x,y,equal_var=False))



from  scipy.stats import ttest_rel
data1 = [0.67, 0.68, 0.78, 0.79, 0.84, 1.20, 1.20, 1.36, 1.45, 1.54]
data2 = [0.89, 0.98, 1.03, 1.12, 1.35, 1.57, 1.71, 1.79, 1.92, 2.05]
print(ttest_rel(data1, data2))