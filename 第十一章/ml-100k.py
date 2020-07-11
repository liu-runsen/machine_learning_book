'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/15
'''

import pandas as pd
df = pd.read_csv('./ml-100k/u.data', names=['userID', 'movieID', 'rating', 'time'], delimiter='	')
print('Rows:', df.shape[0], '; Columns:', df.shape[1], '\n')
######结果如下######
# Rows: 100000 ; Columns: 4

# 查看用户和电影数量
print('No. of Unique Users    :', df.userID.nunique())
print('No. of Unique Movies :', df.movieID.nunique())
print('No. of Unique Ratings  :', df.rating.nunique())

# ####输入如下####
# No. of Unique Users    : 943
# No. of Unique Movies : 1682
# No. of Unique Ratings  : 5

import matplotlib.pyplot as plt

Count_of_Ratings = df.groupby(by=['rating']).agg({'userID': 'count'}).reset_index()
Count_of_Ratings .columns = ['Rating', 'Count']

plt.barh(Count_of_Ratings.Rating, Count_of_Ratings.Count, color='blue')
plt.title('Overall Count of Ratings', fontsize=15)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Rating', fontsize=15)
plt.grid(ls='dotted')
plt.show()



from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# 加载movielens-100k数据集
data = Dataset.load_builtin('ml-100k')

# SVD算法
algo = SVD()

# 5折交叉验证
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

######结果如下######
# Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
#             Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
# RMSE        0.9311  0.9370  0.9320  0.9317  0.9391  0.9342  0.0032
# MAE         0.7350  0.7375  0.7341  0.7342  0.7375  0.7357  0.0015
# Fit time    6.53    7.11    7.23    7.15    3.99    6.40    1.23
# Test time   0.26    0.26    0.25    0.15    0.13    0.21    0.06