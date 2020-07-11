'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''


import numpy as np
from sklearn.preprocessing import Binarizer

x = [ [ 1., 1.,  2.],
      [ 2.,  0.,  0.],
      [ 0.,  1., -1.]]
binarizer=Binarizer().fit(x)
print(binarizer.transform(x))
binarizer = Binarizer(threshold=1.0).fit(x)
print(binarizer.transform(x))


from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
labelList=['yes', 'no', 'no', 'yes','no']
# 将标签矩阵二值化
dummY=lb.fit_transform(labelList)
print("dummY:",dummY)
# 逆过程
yesORno=lb.inverse_transform(dummY)
print("yesOrno:",yesORno)








from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
labelList =np.array(['湖南省','广东省','广西省','湖北省'])
dummY=lb.fit_transform(labelList)
print("dummY:",dummY)
# 逆过程
inverse = lb.inverse_transform(dummY)
print("inverse:",inverse)
