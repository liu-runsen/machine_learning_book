'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/15
'''


import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
data = {'ID':[1,2,3,4,5,6], 'Onion':[1,0,0,1,1,1], 'Potato':[1,1,0,1,1,1],'Burger':[1,1,0,0,1,1],  'Milk':[0,1,1,1,0,1], 'Beer':[0,0,1,0,1,0]}
df = pd.DataFrame(data)
print(df)


frequent_itemsets = apriori(df[['Onion', 'Potato', 'Burger', 'Milk', 'Beer' ]], min_support=0.50, use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
print(rules)


print(rules[(rules['lift'] >1.125)& (rules['confidence']>0.8)])


