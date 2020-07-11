'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0.01,1,100)
plt.plot(x,beta.pdf(x,0.5,0.5),'k-',label='y=Beta(x,0.5,0.5)')
plt.plot(x,beta.pdf(x,5,1),'b-',label='y=Beta(x,5,1)')
plt.plot(x,beta.pdf(x,1,3),'r-',label='y=Beta(x,1,3)')
plt.plot(x,beta.pdf(x,2,2),'g-',label='y=Beta(x,2,2)')
plt.plot(x,beta.pdf(x,2,5),'y-',label='y=Beta(x,2,5)')
plt.legend()
plt.show()
