'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''

from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
n = 10
p = 0.5
k = np.arange(0, 10)
binomial = binom.pmf(k, n, p)
plt.plot(k, binomial)
plt.title('Binomial: n = %i, p=%0.2f' % (n, p), fontsize=15)
plt.xlabel('Number of successes')
plt.ylabel('Probability of sucesses', fontsize=15)
plt.show()


X = np.random.poisson(lam=5, size=10000)  # lam为λ size为k
s= plt.hist(X, bins=15, range=[0, 15], color='g', alpha=0.5)
plt.plot(s[1][0:15], s[0], 'r')
plt.grid()
plt.show()
