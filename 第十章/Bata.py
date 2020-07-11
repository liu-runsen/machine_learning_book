'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/10
'''
from scipy.stats import beta, dirichlet
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100)
a_array = [1, 2, 5]
b_array = [1, 2, 5]
fig, axarr = plt.subplots(len(a_array), len(b_array))
for i, a in enumerate(a_array):
    for j, b in enumerate(b_array):
        axarr[i, j].plot(x, beta.pdf(x, a, b), 'r', lw=1, alpha=0.6, label='a=' + str(a) + ',b=' + str(b))
        axarr[i, j].legend(loc='upper left',fontsize=8)
plt.show()


print(dirichlet.pdf([0.6,0.3,0.1],[3,2,1]))