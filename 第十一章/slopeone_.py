'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/17
'''

from surprise import accuracy
from surprise import Dataset
from surprise import SlopeOne
from surprise.model_selection import train_test_split

# 加载movielens-100k数据集
data = Dataset.load_builtin('ml-100k')

# 训练集和测试集划分
train, test = train_test_split(data, test_size=.15)

# SlopeOne算法
slope = SlopeOne()
slope.fit(train)

# 预测第222用户对第750电影评分
uid = str(222)
iid = str(750)
pred = slope.predict(uid, iid, r_ui=5, verbose=True)
# ######结果如下######
# user: 222
# item: 750
# r_ui = 5.00
# est = 3.97
# {'was_impossible': False}

# 预测第222用户对第750电影评分为3.97

test_pred = slope.test(test)

# RMSE和MAE
print("RMSE: " + str(accuracy.rmse(test_pred, verbose=True)))
print("MAE: " + str(accuracy.mae(test_pred, verbose=True)))

# ######结果如下######
# RMSE: 0.9517
# MAE: 0.7460