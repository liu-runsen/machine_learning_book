'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/15
'''

from efficient_apriori import apriori
# 设置数据集
data = [('牛奶','面包','尿布'),
('可乐','面包', '尿布', '啤酒'),
('牛奶','尿布', '啤酒', '鸡蛋'),
('面包', '牛奶', '尿布', '啤酒'),
('面包', '牛奶', '尿布', '可乐')]

itemsets,rules=apriori(data,min_support=0.5,min_confidence=1)
print(itemsets)
print(rules)



