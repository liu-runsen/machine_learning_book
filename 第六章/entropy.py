'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/21
'''

import numpy as np
import matplotlib.pyplot as plt
eps = 1e-4
p = np.linspace(eps, 1-eps, 100)
h = -(1-p)*np.log2(1-p) - p*np.log2(p)
plt.plot(p,h)
plt.show()
print(max(h))
print(list(h).index(max(h)))
print(p[list(h).index(max(h))])
