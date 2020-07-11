'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/18
'''

'''
计算欧式距离
'''

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
coords1 = [1, 2, 3]
coords2 = [4, 5, 6]
np_c1 = np.array(coords1)
np_c2 = np.array(coords2)
d = np.sqrt(np.sum((np.array(coords1)-np.array((coords2)))**2))
print(d)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter((coords1[0], coords2[0]),
           (coords1[1], coords2[1]),
           (coords1[2], coords2[2]),
           color="k", s=150)
ax.plot((coords1[0], coords2[0]),
        (coords1[1], coords2[1]),
        (coords1[2], coords2[2]),
        color="r")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.text(x=2.5, y=3.5, z=4.0, s='d = {:.2f}'.format(float(d)))
plt.title('Euclidean distance between 3D-coordinates')
plt.show()

