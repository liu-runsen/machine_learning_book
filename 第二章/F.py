'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import f
x= np.linspace(0,3,100)
plt.plot(x,f.pdf(x,20,20),'k-',label='y=f(x,20,20)')
plt.plot(x,f.pdf(x,10,10),'r-',label='y=f(x,10,10)')
plt.plot(x,f.pdf(x,10,5),'g-',label='y=f(x,10,5)')
plt.plot(x,f.pdf(x,10,20),'b-',label='y=f(x,10,20)')
plt.plot(x,f.pdf(x,5,5),'r--',label='y=f(x,5,5)')
plt.plot(x,f.pdf(x,5,10),'g--',label='y=f(x,5,10)')
plt.plot(x,f.pdf(x,5,1),'y-',label='y=f(x,5,1)')
plt.legend()
plt.show()
