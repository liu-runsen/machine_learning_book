'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/3
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_circles
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
X, labels = make_circles(n_samples=200, noise=0.1, factor=0.2)
plt.style.use('ggplot')
df = pd.DataFrame(np.c_[X, labels], columns=['feature1', 'feature2', 'labels'])
plt.subplot(2, 2, 1)
plt.title("原始数据集")
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.subplot(2, 2, 2)
kmeans_labels = KMeans(2, random_state=0).fit_predict(X)
plt.title("K-means划分原始数据集")
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.subplot(2, 2, 3)
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean')
ac_labels = ac.fit_predict(X)
plt.title("层次聚类划分原始数据集")
plt.scatter(X[:, 0], X[:, 1], c=ac_labels)
plt.subplot(2, 2, 4)
dbscan_labels = DBSCAN().fit_predict(X)
plt.title("DBSCAN密度聚类划分原始数据集")
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels)
plt.show()


from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',)
Spectral_labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=Spectral_labels)
plt.title("谱聚类划分原始数据集")
plt.show()




