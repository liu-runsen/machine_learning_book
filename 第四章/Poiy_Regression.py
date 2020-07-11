'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/16
'''

'''
多线性回归实现
'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

# 线性回归
model1 = LinearRegression()
model1.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = model1.predict(xx.reshape(xx.shape[0], 1))
plt.scatter(x=X_train, y=y_train, color='k')
plt.plot(xx, yy, '-g')

# 多项式回归 degree=2 二元二次多项式
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.fit_transform(X_test)
model2 = LinearRegression()
model2.fit(X_train_quadratic, y_train)
xx2 = quadratic_featurizer.transform(xx[:, np.newaxis])
yy2 = model2.predict(xx2)
plt.plot(xx, yy2, '-r')
plt.show()


print('X_train:\n', X_train)
print('X_train_quadratic:\n', X_train_quadratic)
print('X_test:\n', X_test)
print('X_test_quadratic:\n', X_test_quadratic)
print('线性回归R2：', model1.score(X_test, y_test))
print('二次回归R2：', model2.score(X_test_quadratic, y_test))
