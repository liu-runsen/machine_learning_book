'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/2
'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score,calinski_harabasz_score
data=pd.read_csv('Mall_Customers.csv')
# 将年度收入和支出分数作为特征
X=data.iloc[:,[3,4]].values
# 寻找K值
from sklearn.cluster import KMeans
SSE=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)
plt.plot(range(1,11),SSE)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)


print("轮廓系数: " + str(silhouette_score(X, y_kmeans)))
print("CH分数: " + str(calinski_harabasz_score(X, y_kmeans)))

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='magenta',label='Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='yellow',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='burlywood',label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='red',label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

sns.lmplot(x='Age', y='Spending Score (1-100)', data=data,fit_reg=True,hue='Gender')
plt.show()

data.sort_values(['Age'])
plt.figure(figsize=(10,8))
plt.bar(data['Age'],data['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.show()

label_encoder=LabelEncoder()
integer_encoded=label_encoder.fit_transform(data.iloc[:,1].values)
data['Gender']=integer_encoded
hm=sns.heatmap(data.iloc[:,1:5].corr(), annot = True, linewidths=.5, cmap='Blues')
hm.set_title(label='Heatmap of dataset', fontsize=20)
plt.show()

