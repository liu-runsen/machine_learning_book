'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/16
'''

'''
线性回归的实现
'''

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,30,50)
y = x+ 2*np.random.rand(50)
plt.figure(figsize=(10,8))
plt.scatter(x,y)

from sklearn.linear_model import LinearRegression
# 初始化模型
model = LinearRegression()
# 将行变列，得到x坐标
x1 = x.reshape(-1,1)
# 将行变列  得到y坐标
y1 = y.reshape(-1,1)
# 训练数据
model.fit(x1,y1)
# 预测下x=40 ，y的值
plt.figure(figsize=(12,8))
plt.scatter(x,y)
x_test = np.linspace(0,40).reshape(-1,1)
plt.plot(x_test,model.predict(x_test))
plt.show()


# 斜率
print(model.coef_)
# 截距
print(model.intercept_)

print(model.predict(np.array(40).reshape(1,-1)))