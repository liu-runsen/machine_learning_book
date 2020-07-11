'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/26
'''

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
rng = np.random.RandomState(0)
x = 10 * rng.rand(200)
def ture_model(x, sigma=0.3):
    y1 = np.sin(0.5 * x )
    y2 = np.cos(0.5 * x )
    noise = sigma * rng.randn(len(x))
    return y1 + y2 + noise
y = ture_model(x)
plt.figure(figsize=(10,8))
plt.scatter(x,y,c = 'r')
plt.show()



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)
xfit = np.linspace(0, 10, 1000)
yfit = forest.predict(xfit[:, None])
ytrue = ture_model(xfit, sigma=0)
plt.figure(figsize=(10,8))
plt.scatter(x,y)
plt.plot(xfit, yfit, '-r',label="预测值")
plt.plot(xfit, ytrue, '-k', alpha=0.5,label="真实值")
plt.legend()
plt.show()
print(mean_squared_error(ytrue,yfit))