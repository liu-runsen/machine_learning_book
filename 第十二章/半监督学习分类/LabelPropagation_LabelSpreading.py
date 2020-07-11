'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/24
'''

import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
iris = datasets.load_iris()
X = iris.data.copy()
y = iris.target.copy()
y[np.random.choice([True, False], len(y))] = -1
print(y)
lp = LabelPropagation()
lp.fit(X, y)
preds = lp.predict(X)
print((preds == iris.target).mean())
# 0.9933333333333333
ls = LabelSpreading()
ls.fit(X, y)
print((ls.predict(X)== iris.target).mean())
# 0.9866666666666667