'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/28
'''

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
N = 30
np.random.seed(0)
x = np.sort(np.random.uniform(0, 2*np.pi, N), axis=0)
y = 2*np.sin(x) + 0.2*np.random.randn(N)
x = x.reshape(-1, 1)
print('SVR - RBF')
svr_rbf = svm.SVR(kernel='rbf', gamma=0.2, C=100)
svr_rbf.fit(x, y)
print(svr_rbf.score(x,y))
print('SVR - Linear')
svr_linear = svm.SVR(kernel='linear', C=100)
svr_linear.fit(x, y)
print(svr_linear.score(x,y))
print('SVR - Polynomial')
svr_poly = svm.SVR(kernel='poly', degree=3, C=100)
svr_poly.fit(x, y)
print(svr_poly.score(x,y))
x_test = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_rbf = svr_rbf.predict(x_test)
y_linear = svr_linear.predict(x_test)
y_poly = svr_poly.predict(x_test)
plt.scatter(x, y, c='b')
plt.show()
plt.figure(figsize=(8, 10), facecolor='w')
plt.plot(x_test, y_rbf, 'r-', linewidth=2, label='RBF Kernel')
plt.plot(x_test, y_linear, 'g-', linewidth=2, label='Linear Kernel')
plt.plot(x_test, y_poly, 'b-', linewidth=2, label='Polynomial Kernel')
plt.plot(x, y, 'mo', markersize=6, markeredgecolor='k')
plt.scatter(x[svr_rbf.support_], y[svr_rbf.support_], s=200, c='r', marker='*', edgecolors='k', label='Support Vectors', zorder=10)
plt.legend(loc='lower left', fontsize=12)
plt.title('SVR', fontsize=15)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(b=True, ls=':')
plt.tight_layout(2)
plt.show()
