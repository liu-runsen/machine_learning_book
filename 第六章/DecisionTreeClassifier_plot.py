'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/24
'''

import numpy as np
import pydotplus
from matplotlib import  pyplot as plt
from sklearn.tree import DecisionTreeClassifier,export_graphviz
X = np.array([[2, 2],
              [2, 1],
              [2, 3],
              [1, 2],
              [1, 1],
              [3, 3]])

y = np.array([0, 1, 1, 1, 0, 1])
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.figure(figsize=(8, 8))
# 绘制标签
for x1, x2, label in zip(X[:, 0], X[:, 1], y):
    plt.text(x1, x2, str(label), fontsize=40, color='g',
             ha='center', va='center')
plt.grid(None)
plt.xlim((0, 3.5))
plt.ylim((0, 3.5))
plt.xlabel('x1', size=20)
plt.ylabel('x2', size=20)
plt.title('Data', size=24)
plt.show()
dot_tree = DecisionTreeClassifier()
print(dot_tree)
dot_tree.fit(X, y)
print(dot_tree.score(X, y))
dot_data = export_graphviz(dot_tree, out_file=None,
                                feature_names=['x1', 'x2'],
                                class_names=['0', '1'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
with open('dot.png', 'wb') as f:
    f.write(graph.create_png())
graph.write_pdf('dot.pdf')


