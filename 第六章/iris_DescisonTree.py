'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/24
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pydotplus
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
iris = load_iris()
x = pd.DataFrame(iris.data)
y = iris.target
# 为了可视化，仅使用前两列特征
x = x[[0,1]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
# 决策树参数估计
# min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
# min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则进行分支；否则，不进行分支
# DecisionTreeClassifier默认使用criterion="gini"，也就是CART算法，这里使用的entropy信息熵
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
print('训练集正确率：', accuracy_score(y_train, y_train_pred))
y_test_hat = model.predict(x_test)
print('测试集正确率：', accuracy_score(y_test, y_test_hat))
with open('iris.dot', 'w') as f:
    tree.export_graphviz(model, out_file=f, feature_names=iris_feature_E[0:2], class_names=iris_class,
                         filled=True, rounded=True, special_characters=True)
tree.export_graphviz(model, out_file='iris.dot', feature_names=iris_feature_E[0:2], class_names=iris_class,
                     filled=True, rounded=True, special_characters=True)
tree.export_graphviz(model, out_file='iris.dot')
# 输出为pdf格式
dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E[0:2], class_names=iris_class,
                                filled=True, rounded=True, special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('iris.pdf')
f = open('iris.png', 'wb')
f.write(graph.create_png())
f.close()
# 画图
N, M = 50, 50
# 横纵各采样多少个值
x1_min, x2_min = x.min()
x1_max, x2_max = x.max()
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)
# 生成网格采样点
x_show = np.stack((x1.flat, x2.flat), axis=1)
# 测试点
y_show_hat = model.predict(x_show).reshape(x1.shape)
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
plt.scatter(x_test[0], x_test[1], c=y_test.ravel(), edgecolors='k', s=100, zorder=10, cmap=cm_dark, marker='*')  # 测试数据
plt.scatter(x[0], x[1], c=y.ravel(), edgecolors='k', s=20, cmap=cm_dark)  # 全部数据
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(b=True, ls=':', color='#606060')
plt.title('鸢尾花数据的决策树分类', fontsize=15)
plt.show()





# 过拟合：错误率
depth = np.arange(1, 15)
err_train_list = []
err_test_list = []
clf = DecisionTreeClassifier(criterion='entropy')
for d in depth:
    clf.set_params(max_depth=d)
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    err_train = 1 - accuracy_score(y_train, y_train_pred)
    err_train_list.append(err_train)
    y_test_pred = clf.predict(x_test)
    err_test = 1 - accuracy_score(y_test, y_test_pred)
    err_test_list.append(err_test)
    print(d, ' 测试集错误率: %.2f%%' % (100 * err_test))
plt.figure(facecolor='w')
plt.plot(depth, err_test_list, 'ro-', markeredgecolor='k', lw=2, label='测试集错误率')
plt.plot(depth, err_train_list, 'go-', markeredgecolor='k', lw=2, label='训练集错误率')
plt.xlabel('决策树深度', fontsize=13)
plt.ylabel('错误率', fontsize=13)
plt.legend(loc='lower left', fontsize=13)
plt.title('决策树深度与过拟合', fontsize=15)
plt.grid(b=True, ls=':', color='#606060')
plt.show()