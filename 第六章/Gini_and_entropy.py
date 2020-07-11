'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/23
'''

import numpy as np
import matplotlib.pyplot as plt

eps = 1e-4
p = np.linspace(eps, 1-eps, 100)
h = -(1-p)*np.log2(1-p) - p*np.log2(p)
gini = 2*(1-p)*p
plt.plot(p, gini, 'r-', lw=3,label = "Gini")
plt.plot(p, h/2, 'g-', lw=3,label = "entropy")
plt.title('Gini/Entropy', fontsize=16)
plt.legend()
plt.show()
