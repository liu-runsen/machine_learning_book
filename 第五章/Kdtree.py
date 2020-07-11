'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/19
'''
import numpy as np
from sklearn.neighbors import KDTree
from collections import namedtuple
from operator import itemgetter
from pprint import pformat

# 节点类,（namedtuple）Node中包含样本点和左右叶子节点
class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))

# 构造kd树
def kdtree(point_list, depth=0):
    try:
        # 假设所有点都具有相同的维度
        k = len(point_list[0])
    # 如果不是point_list返回None
    except IndexError as e:
        return None
    # 根据深度选择轴，以便轴循环所有有效值
    axis = depth % k

    # 排序点列表并选择中位数作为主元素
    point_list.sort(key=itemgetter(axis))
    # 向下取整
    median = len(point_list) // 2

    # 创建节点并构建子树
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1))

if __name__ == '__main__':
    point_list = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = kdtree(point_list)
    print(tree)
    # sklearn实现KD树
    tree = KDTree(np.array(point_list), leaf_size=2)
    # ind：最近的1个邻居的索引
    # dist：距离最近的1个邻居
    # np.array([2.1,3.1]):搜索点
    dist, ind = tree.query([np.array([2.1, 3.1])], k=1)
    print('ind:', ind)
    print('dist:', dist)
