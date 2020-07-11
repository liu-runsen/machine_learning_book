'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/19
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection  import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
#读取鸢尾花数据集
iris = load_iris()
x = iris.data
y = iris.target
k_range = range(1, 20)
k_score = []
#循环，取k=1到k=20，查看正确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #cv参数决定数据集划分比例，这里是按照4:1划分训练集和测试集
    scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
    k_score.append(np.around(scores.mean(),3))
#画图，x轴为k值，y值为正确率
plt.plot(k_range, k_score)
plt.xticks(np.linspace(1,20,20))
plt.xlabel('Value of K for KNN')
plt.ylabel('score')
plt.show()
print(k_score)
print("最终的最佳K值:{}".format(int(k_score.index(max(k_score))) + 1 ))
print("最终最佳准确率：{}".format(max(k_score)))

