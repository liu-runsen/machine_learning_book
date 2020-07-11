'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''
from sklearn.datasets import load_iris
iris = load_iris()


import pandas as pd
print(pd.DataFrame(iris.data[:5],columns = iris['feature_names']))


from sklearn import datasets
iris = datasets.load_iris()
# 方差选择法，返回值为特征选择后的数据
from sklearn.feature_selection import VarianceThreshold
# 参数threshold为方差的阈值，这里指定3
vardata = VarianceThreshold(threshold=3).fit_transform(iris.data)
print(vardata[:5])


import numpy as np
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
#输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。
#在此定义为计算相关系数
f = lambda X, Y:np.array(list(map(lambda x:pearsonr(x, Y)[0], X.T))).T
#参数k为选择的特征个数
print(SelectKBest(f,k=2).fit_transform(iris.data, iris.target)[:5])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#选择K个最好的特征，返回选择特征后的数据
print(SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)[:5])


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型，这里选择逻辑回归
#参数n_features_ to_select为选择的特征个数
print(RFE(estimator=LogisticRegression(), n_features_to_select=3).fit_transform(iris.data, iris.target)[:5])


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
# 基于L1惩罚项的逻辑回归作为基模型的特征选择
print("基于l1惩罚项的逻辑回归：\n"+ str(SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)[:5]) + '\n')
# 默认是L2惩罚项 penalty=“l2”,
print("基于l2惩罚项的逻辑回归：\n"+ str(SelectFromModel(LogisticRegression(C=0.1)).fit_transform(iris.data, iris.target)[:5]))


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
# 基于L1惩罚项的逻辑回归作为基模型的特征选择
print("基于l1惩罚项的逻辑回归：\n" + str(SelectFromModel(LogisticRegression(penalty='l1', C=0.1)).fit_transform(iris.data, iris.target)[:5]) + '\n')
# 默认是L2惩罚项 penalty=“l2”,
print("基于l2惩罚项的逻辑回归：\n" + str(SelectFromModel(LogisticRegression(C=0.1)).fit_transform(iris.data, iris.target)[:5]))
