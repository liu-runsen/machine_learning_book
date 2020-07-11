'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/24
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def distEclud(vecA, vecB):
    '''
    输入：向量A和B
    输出：A和B间的欧式距离
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def newCent(L):
    '''
    输入：有标签数据集L
    输出：根据L确定初始聚类中心
    '''
    centroids = []
    label_list = np.unique(L[:, -1])
    for i in label_list:
        L_i = L[(L[:, -1]) == i]
        cent_i = np.mean(L_i, 0)
        centroids.append(cent_i[:-1])
    return np.array(centroids)

def semi_kMeans(L, U, distMeas=distEclud, initial_centriod=newCent):
    '''
    输入：有标签数据集L（最后一列为类别标签）、无标签数据集U（无类别标签）
    输出：聚类结果
    '''
    dataSet = np.vstack((L[:, :-1], U))  # 合并L和U
    label_list = np.unique(L[:, -1])
    k = len(label_list)  # L中类别个数
    m = np.shape(dataSet)[0]
    clusterAssment = np.zeros(m)  # 初始化样本的分配
    centroids = initial_centriod(L)  # 确定初始聚类中心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 将每个样本分配给最近的聚类中心
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i] != minIndex: clusterChanged = True
            clusterAssment[i] = minIndex
    return clusterAssment

data = pd.read_excel("数据.xlsx")

train = data.iloc[:35, :]
test = data.iloc[35:, :]
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train.iloc[:, -1].values)
train.loc[:, "类别"] = integer_encoded
print(label_encoder.classes_)

L = train.iloc[:, 0:4].values
U = test.iloc[:, 0:3].values
clusterResult = semi_kMeans(L, U).astype(np.int)
Result = list(label_encoder.inverse_transform(clusterResult))
print(Result[-10:])
data = data.drop(columns=['类别'])
data['类别'] = Result
data.to_excel("Result.xlsx")
