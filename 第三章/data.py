'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
center=[[1,1],[-1,-1],[1,-1]]
X,labels=make_blobs(n_samples=200,centers=center,n_features=2, cluster_std=0.3)
# np.c_方法将X和labels变成DataFrame中的列
df = pd.DataFrame(np.c_[X,labels],columns = ['feature1','feature2','labels'])
# matplotlib常用colormap分别有'jet','rainbow','hsv'
df.plot.scatter('feature1','feature2', s = 100, c = list(df['labels']), cmap = 'rainbow',colorbar = False, alpha = 0.8,title = 'dataset by make_blobs')
plt.show()


from sklearn.datasets.samples_generator import make_classification
X,labels=make_classification(n_samples=300,n_features=2,n_classes = 2,n_redundant=0)
# 加入噪声数据
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
df = pd.DataFrame(np.c_[X,labels],columns = ['feature1','feature2','labels'])
df.plot.scatter('feature1','feature2', s = 100, c = list(df['labels']),cmap = 'rainbow',colorbar = False, alpha = 0.8,title = 'dataset by make_classification')
plt.show()


from sklearn.datasets.samples_generator import make_circles
X,labels=make_circles(n_samples=200,noise=0.2,factor=0.2)
df = pd.DataFrame(np.c_[X,labels],columns = ['feature1','feature2','labels'])
df.plot.scatter('feature1','feature2', s = 100, c = list(df['labels']),
cmap = 'rainbow',colorbar = False, alpha = 0.8,title = 'dataset by make_circles')
plt.show()


from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_moons
x1,y1=make_moons(n_samples=1000,noise=0.1)
plt.title('make_moons function example')
plt.scatter(x1[:,0],x1[:,1],marker='o',c=y1)
plt.show()


from sklearn.datasets.samples_generator import make_regression
X,Y,coef = make_regression(n_samples=100, n_features=1, bias=5, tail_strength= 0, noise= 1, shuffle=True, coef=True, random_state=None)
print(coef) #49.08950060982939
df = pd.DataFrame(np.c_[X,Y],columns = ['x','y'])
df.plot('x','y',kind = 'scatter',s = 50,c = 'm',edgecolor = 'k')
plt.show()
