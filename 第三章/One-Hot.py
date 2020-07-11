'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''

import pandas as pd
sex = ["male","female"]
print(pd.get_dummies(sex))
print(type(pd.get_dummies(sex)))


from sklearn.preprocessing import OneHotEncoder
sex = [["male"],
	 ["female"]]
onehot = OneHotEncoder()
print(onehot.fit_transform(sex)).toarray()
print(onehot.categories_)




x = [[0, 0, 3],
     [1, 1, 0],
     [0, 2, 1],
     [1, 0, 2]]
onehot = OneHotEncoder()
print(onehot.fit_transform(x).toarray())
print(onehot.categories_)


a = [["Runsen", "Runsen", "wangwu"],
     ["Zhangsan", "Zhangsan", "Runsen"],
     ["Runsen", "Lisi", "Zhangsan"],
     ["Zhangsan","Runsen", "Lisi"]]
onehot = OneHotEncoder()
print(onehot.fit_transform(a).toarray())
print(onehot.categories_)



import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
print(X)

# PolynomialFeatures进行多项式特征，设置多项式阶数为２
poly = PolynomialFeatures(2)
print(poly.fit_transform(X))


poly = PolynomialFeatures(include_bias=False)
print(poly.fit_transform(X))


poly = PolynomialFeatures(interaction_only=True)
print(poly.fit_transform(X))
