'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/20
'''


import numpy as np
from sklearn.naive_bayes import BernoulliNB
X = np.array([[1,2,3,4],[1,3,4,4],[2,4,5,5]])
y = np.array([1,1,2])
# binarize阈值为3
# 二值化后X如下
# [[0,0,0,1],   类别1
#  [0,0,1,1],   类别1
#  [0,1,1,1]]   类别2
clf = BernoulliNB(binarize = 3.0)
clf.fit(X,y)
# 按类别顺序输出其对应的个数
print(clf.class_count_)
# 各类别各特征值之和，
print(clf.feature_count_)
# 伯努利分布的P值
print(clf.feature_log_prob_)

# [[-1.09861229 -1.09861229 -0.69314718 -0.40546511]
#  [-0.91629073 -0.51082562 -0.51082562 -0.51082562]]

print([np.log((2+2)/(2+2*2))*0+np.log((0 + 2 )/(2+    2 * 2))*1,
 np.log((2+2)/(2+2*2))*0+np.log((0 + 2 )/(2+    2 * 2))*1,
 np.log((1+2)/(2+2*2))*0+np.log((1 + 2 )/(2+    2 * 2))*1,
 np.log((0+2)/(2+2*2))*0+np.log((2 + 2 )/(2+    2 * 2))*1])
#          ↑A                      ↑B  ↑α   ↑样本  ↑α  ↑类别数




# #说明:

# A列为2,2,1,0  类别1中0个数
# B列为0,0,1,2  类别2中1个数
# 样本:为当前类别样本数

