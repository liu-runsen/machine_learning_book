'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/2
'''

from sklearn  import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score,calinski_harabasz_score
x, y = datasets.make_blobs(400, n_features=2, centers=4, random_state=0)
model = KMeans(n_clusters=4)
model.fit(x)
y_pred = model.predict(x)
print(" 调整兰德系数: " + str(adjusted_rand_score(y, y_pred)))
print(" 轮廓系数: " + str(silhouette_score(x, y_pred)))
print(" CH分数: " + str(calinski_harabasz_score(x, y_pred)))

