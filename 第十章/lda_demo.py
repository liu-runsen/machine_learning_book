'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/12
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import lda
from lda.datasets import load_reuters,load_reuters_vocab,load_reuters_titles
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
from pprint import pprint
# 文档矩阵
X = load_reuters()
print(("shape: {}\n".format(X.shape)))
print((X[:10, :10]))

# 加载词
vocab =load_reuters_vocab()
print(("type(vocab): {}".format(type(vocab))))
print(("len(vocab): {}\n".format(len(vocab))))
pprint((vocab[:10]))

# 加载标题
titles = load_reuters_titles()
print(("标题的类型：type(titles): {}".format(type(titles))))
print(("标题的数目：len(titles): {}\n".format(len(titles))))
print(titles[:10])

print('开始建立LDA主题模型')
model = lda.LDA(n_topics=20, n_iter=800, random_state=1)
model.fit(X)

# 主题到词
topic_word = model.topic_word_
print(("主题到词:type(topic_word): {}".format(type(topic_word))))
print(("主题到词:shape: {}".format(topic_word.shape)))
print((vocab[:5]))

# 文档到主题
doc_topic = model.doc_topic_
print(("shape: {}".format(doc_topic.shape)))
for i in range(5):
    topic_most_pr = doc_topic[i].argmax()
    print(("文档: {} 主题: {} value: {}".format(i, topic_most_pr, doc_topic[i][topic_most_pr])))

# 主题到词
plt.figure(figsize=(10, 8))
for i, k in enumerate([0, 5, 9, 14, 19]):
    ax = plt.subplot(5, 1, i+1)
    ax.plot(topic_word[k, :], 'r-')
    ax.set_xlim(-50, 4350)   # [0,4258]
    ax.set_ylim(0, 0.08)
    ax.set_ylabel("概率")
    ax.set_title("主题 {}".format(k))
plt.xlabel("主题的词分布", fontsize=14)
plt.tight_layout()
plt.show()

# 文档到主题
plt.figure(figsize=(10, 8))
for i, k in enumerate([1, 3, 4, 8, 9]):
    ax = plt.subplot(5, 1, i+1)
    ax.stem(doc_topic[k, :], linefmt='g-', markerfmt='ro')
    ax.set_xlim(-1,20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("概率")
    ax.set_title("文档 {}".format(k))
plt.xlabel("文档的主题分布", fontsize=14)
plt.subplots_adjust(top=0.9)
plt.show()
