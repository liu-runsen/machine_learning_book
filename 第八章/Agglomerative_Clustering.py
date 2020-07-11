'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/5
'''

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
np.random.seed(0)
labels = ['特征01', '特征02', '特征03', '特征04', '特征05']
X = np.random.random_sample([5, 5]) * 10
# 层次聚类树
df = pd.DataFrame(X, columns=labels)
print(df)
# 计算距离关联矩阵，两两样本间的欧式距离
row_dist = pd.DataFrame(squareform(pdist(df,metric='euclidean')),columns=labels,index=labels)
print (row_dist)
linkages = ["ward", "complete", "average"]
for index, i in enumerate(linkages):
    row_clusters = linkage(pdist(df, metric='euclidean'), method=i)
    print('{}cluster'.format(str(i)))
    print(pd.DataFrame(row_clusters, columns=['row label1', 'row label2', 'distance', 'no. of items in clust.'],index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]))
    # 层次聚类树
    plt.subplot(3,1,index+1)
    plt.ylabel('{}聚类'.format(i))
    row_dendr = dendrogram(row_clusters, labels=labels)
    ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage=i)
    pred_labels = ac.fit_predict(X)
    print('{}cluster labels:{}\n'.format(str(i),pred_labels))
plt.show()
