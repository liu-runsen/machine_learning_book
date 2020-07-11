'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/18
'''

'''
KNN算法实现
'''
import numpy as np
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 导入iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)

def euc_dis(instance1, instance2):
    """
    计算两个样本instance1和instance2之间的欧式距离
    instance1: 第一个样本， array型
    instance2: 第二个样本， array型
    """
    dist = np.sqrt(sum((instance1 - instance2) ** 2))
    return dist

def knn_classify(X, y, testInstance, k):
    """
    给定一个测试数据testInstance, 通过KNN算法来预测它的标签。
    X: 训练数据的特征
    y: 训练数据的标签
    testInstance: 测试数据，这里假定一个测试数据 array型
    k: 选择多少个neighbors?
    """
    # 返回testInstance的预测标签 = {0,1,2}
    distances = [euc_dis(x, testInstance) for x in X]
    # 排序
    kneighbors = np.argsort(distances)[:k]
    # count是一个字典
    count = Counter(y[kneighbors])
    # count.most_common()[0][0])是票数最多的
    return count.most_common()[0][0]

# 预测结果 iris是典型的三分类数据集 这里的K指定为3
predictions = [knn_classify(X_train, y_train, data, 3) for data in X_test]
print(predictions[:5])
correct = np.count_nonzero((predictions == y_test) == True)
print(correct)
clf  = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print("sklearn KNN-model's Accuracy is: %.3f" %(accuracy_score(y_test, clf.predict(X_test))))
print("My KNN-model's Accuracy is: %.3f" % (correct / len(X_test)))
