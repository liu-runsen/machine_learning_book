'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/18
'''

'''
余弦距离
'''

import pandas as pd
import numpy as np
data = pd.DataFrame({'one': [4, np.nan, 2, np.nan],
                     'two': [np.nan, 4, np.nan, 5],
                     'three': [5, np.nan, 2, np.nan],
                     'four': [3, 4, np.nan, 3],
                     'five': [5, np.nan, 1, np.nan],
                     'six': [np.nan, 5, np.nan, 5],
                     'seven': [np.nan, np.nan, np.nan, 4]},
                     index = list('ABCD'))
print(data)
#
from sklearn.metrics.pairwise import cosine_similarity
# sim_AB = cosine_similarity(data.loc['A', :].fillna(0).values.reshape(1, -1),
#                            data.loc['B', :].fillna(0).values.reshape(1, -1))
# sim_AC = cosine_similarity(data.loc['A', :].fillna(0).values.reshape(1, -1),
#        					   data.loc['C', :].fillna(0).values.reshape(1, -1))
# print(sim_AB)
# print(sim_AC)

data_center = data.apply(lambda x: x-x.mean(), axis=1)
print(data_center)


sim_AB = cosine_similarity(data_center.loc['A', :].fillna(0).values.reshape(1, -1),
                          data_center.loc['B', :].fillna(0).values.reshape(1, -1))
sim_AC = cosine_similarity(data_center.loc['A', :].fillna(0).values.reshape(1, -1),
                          data_center.loc['C', :].fillna(0).values.reshape(1, -1))
print(sim_AB)
print(sim_AC)


sim_AD = cosine_similarity(data_center.loc['A', :].fillna(0).values.reshape(1, -1),
                           data_center.loc['D', :].fillna(0).values.reshape(1, -1))
print(sim_AD)
print((sim_AD*data.loc['D', 'two'] + sim_AB*data.loc['B', 'two'])/(sim_AD + sim_AB)
)