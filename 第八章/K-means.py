'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/1
'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data1 = pd.DataFrame({'X':np.random.randint(1,50,100),'Y':np.random.randint(1,50,100)})
data = pd.concat([data1 + 50, data1])
plt.style.use('ggplot')
plt.scatter(data.X, data.Y)
plt.show()

# 导入kmeans算法
from sklearn.cluster import KMeans
# 预测分为2类，
y_pred = KMeans(n_clusters=2).fit_predict(data)
# 用color分开出来
plt.scatter(data.X,data.Y,c=y_pred)
plt.show()

from sklearn.metrics import calinski_harabasz_score
print(calinski_harabasz_score(data,y_pred))
