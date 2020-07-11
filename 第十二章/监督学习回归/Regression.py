'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/19
'''

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False



x = np.linspace(0, 30, 50)
y = x + 2 * np.random.rand(50)


def y(x1, x2):
    y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2)
    return y


def load_data():
    x1_train = np.linspace(0, 50, 500)
    x2_train = np.linspace(-10, 10, 500)
    data_train = np.array([[x1, x2, y(x1, x2) + (np.random.random(1) - 0.5)] for x1, x2 in zip(x1_train, x2_train)])
    x1_test = np.linspace(0, 50, 100) + 0.5 * np.random.random(100)
    x2_test = np.linspace(-10, 10, 100) + 0.5 * np.random.random(100)
    data_test = np.array([[x1, x2, y(x1, x2)] for x1, x2 in zip(x1_test, x2_test)])
    return data_train, data_test


def try_different_method(clf,name):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print('{}score: {:.3f}'.format(name,score))

    # result = clf.predict(x_test)
    # plt.figure()
    # plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    # plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    # plt.title('{}   score: {:.3f}'.format(name,score))
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    train, test = load_data()
    # 数据 前两列是x1,x2 第三列是y
    x_train, y_train = train[:, :2], train[:, 2]
    x_test, y_test = test[:, :2], test[:, 2]

    # 线性回归
    try_different_method(LinearRegression(),"线性回归")
    # Lasso回归
    try_different_method(Lasso(), "Lasso回归")
    # 岭回归
    try_different_method(Ridge(), "岭回归")
    # ElasticNet回归
    try_different_method(ElasticNet(), "ElasticNet回归")
    # SVR
    try_different_method(SVR(), "SVR")
    # 回归树
    try_different_method(DecisionTreeRegressor(), "回归树")
    # KNN
    try_different_method(KNeighborsRegressor(),"KNN")
    # 集成学习Bagging
    try_different_method(BaggingRegressor(), "集成学习Bagging")
    # 集成学习ADA
    try_different_method(AdaBoostRegressor(), "集成学习ADA")
    # 集成学习梯度提升
    try_different_method(GradientBoostingRegressor(), "集成学习梯度提升")
    # 随机森林
    try_different_method(RandomForestRegressor(),"随机森林")
    print('down')
