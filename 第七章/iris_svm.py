'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/27
'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
from time import time

iris_feature = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
iris = load_iris()
X = pd.DataFrame(iris.data)[[2,3]]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.4)

def SVN_linear(X,y):
    t = time()
    clf = svm.SVC(C=3, kernel='linear')
    clf.fit(x_train, y_train.ravel())
    print('\n耗时：%f秒' % (time() - t))
    print("鸢尾花SVM线性核二特征分类准确度%f" %clf.score(x_train, y_train))
    print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
    print(clf.score(x_test, y_test))
    print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))
    plot(X,y,clf,"鸢尾花SVM线性核二特征分类")

def SVN_poly(X,y):
    t = time()
    clf = svm.SVC(C=3, kernel='poly', degree=3)
    clf.fit(x_train, y_train.ravel())
    print('\n耗时：%f秒' % (time() - t))
    print("鸢尾花SVM多线式核二特征分类准确度%f" %clf.score(x_train, y_train))
    print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
    print(clf.score(x_test, y_test))
    print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))
    plot(X,y,clf,"鸢尾花SVM多线式核二特征分类")

def SVN_rbf(X,y):
    t = time()
    clf = svm.SVC(C=10, gamma=1, kernel='rbf', decision_function_shape='ovo')
    grid_search = GridSearchCV(clf, param_grid={'gamma': np.logspace(-2, 2, 10), 'C': np.logspace(-2, 2, 10)}, cv=3)
    grid_search.fit(x_train, y_train.ravel())
    print('\n耗时：%4f秒' % (time() - t))
    print('最优参数：', grid_search.best_params_)
    print("鸢尾花SVM高斯核二特征分类准确度%2f" % grid_search.score(x_train, y_train))
    print('训练集准确率：', accuracy_score(y_train, grid_search.predict(x_train)))
    print(grid_search.score(x_test, y_test))
    print('测试集准确率：', accuracy_score(y_test, grid_search.predict(x_test)))
    plot(X, y,grid_search, "鸢尾花SVM高斯核二特征分类")

def plot(X,y,clf,title):
    x1_min, x2_min = X.min()
    x1_max, x2_max = X.max()
    x1, x2 = np.mgrid[x1_min:x1_max:300j, x2_min:x2_max:300j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    grid_hat = clf.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(X[2], X[3], c=y, edgecolors='k', s=50, cmap=cm_dark)
    plt.scatter(x_test[2], x_test[3], s=120, facecolors='none', zorder=10)
    plt.xlabel(iris_feature[2], fontsize=13)
    plt.ylabel(iris_feature[3], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(label=title, fontsize=16)
    plt.grid(b=True)
    plt.tight_layout(pad=1.5)
    plt.show()

if __name__ == '__main__':
    SVN_linear(X, y)
    SVN_poly(X, y)
    SVN_rbf(X, y)
