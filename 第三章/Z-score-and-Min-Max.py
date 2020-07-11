'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''

import numpy as np
from sklearn import preprocessing

x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

# 将每一列特征标准化为标准正态分布
x_scale = preprocessing.scale(x)
print(x_scale)
print(x_scale.mean(axis = 0))
print(x_scale.std(axis=0))

x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

scaler = preprocessing.StandardScaler().fit(x)
print(scaler.transform(x))
new_x = [[-1., 1., 0.]]
print(scaler.transform(new_x))
print(scaler.transform(new_x).mean(axis = 0))
print(scaler.transform(new_x).std(axis=0))

from sklearn.preprocessing import MinMaxScaler

x_train = np.array([[1, -1, 2],
                    [2, 0, 0],
                    [0, 1, -1]])

min_max_scaler = MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
print(x_train_minmax)


# 现在又来了一组新的样本，也想得到相同的转换
x_test = np.array([[-3., -1., 4.]])
x_test_minmax = min_max_scaler.transform(x_test)
print(x_test_minmax)
