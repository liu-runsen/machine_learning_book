'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''

from sklearn.feature_extraction import DictVectorizer

# 定义一个字典列表 用来表示多个数据样本
dicts = [
    {"name": "Runsen", "age": 20},
    {"name": "Zhangsan", "age": 21},
    {"name": "Lisi", "age": 22},
]

# 初始化字典特征抽取器
vec = DictVectorizer()
data = vec.fit_transform(dicts).toarray()
# 查看提取后的特征值
print(data)

from sklearn.feature_extraction.text import CountVectorizer
corpus = ['Today is Friday. The weather is fine',
          'Because of the fine weather today, I decided to go out',
          'It rained on Friday afternoon and I had to go home ' ]
# 设置英语的常用停用词
vectorizer = CountVectorizer(stop_words = 'english')
# 转化为稀疏矩阵
print(vectorizer.fit_transform(corpus).todense())
# 查看每行column特征的含义
print(vectorizer.vocabulary_)


import jieba
seg_list = jieba.cut("我叫小明，是一名大学生")
print("/".join(seg_list))

import jieba

corpus = ['我叫小明，是一名大学生',
          '小明是一名大学生',
          '小明的同学小红也是一名大学生']

cutcorpus = ["/ ".join(jieba.cut(x))
for x in corpus]
print(cutcorpus)

from sklearn.feature_extraction.text import CountVectorizer
# 加载停用词
vectorizer = CountVectorizer("我","是")
counts = vectorizer.fit_transform(cutcorpus).todense()
print(counts)
print(vectorizer.vocabulary_)


# 导入欧式距离
from sklearn.metrics.pairwise import  euclidean_distances
vectorizer = CountVectorizer()
for x,y in [[0,1],[0,2],[1,2]]:
    dist = euclidean_distances(counts[x],counts[y])
    print('文档{}与文档{}的距离{}'.format(x,y,dist))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['Today is Friday. The weather is fine',
          'Because of the fine weather today, I decided to go out',
          'It rained on Friday afternoon and I had to go home ']

words = CountVectorizer().fit_transform(corpus)
tfidf = TfidfTransformer().fit_transform(words)

print(words.todense())
print(tfidf)
