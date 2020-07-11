'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/11
'''

from collections import defaultdict
text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# 创建一组常用词
stoplist = set('for a of the and to in'.split(' '))
# 将每个文档小写，用空格分隔，并筛选出停用词
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# 计算字频
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# 只保留出现多次的单词
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
print(processed_corpus)


from gensim import corpora
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
# Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...)
# 对应的词频
print(dictionary.token2id)
# {'minors': 11, 'graph': 10, 'system': 5, 'trees': 9, 'eps': 8, 'computer': 0,'survey': 4, 'user': 7, 'human': 1, 'time': 6, 'interface': 2, 'response': 3}
# 标记化文档转换为矢量doc2bow方法
corpus = [dictionary.doc2bow(text) for text in processed_corpus]
print(corpus[0])
# [(0, 1), (1, 1), (2, 1)]
# (0, 1)代表human，(1, 1)代表interface，(2, 1)代表computer
print(len(corpus)) #9


from gensim import models
# 模型对象的初始化
tfidf = models.TfidfModel(corpus)
print(tfidf)



# [(0, 1), (1, 1)] 代表'human', 'interface'
doc_bow = [(0, 1), (1, 1)]
# TfIdf实值权
print(tfidf[doc_bow])
# gensim训练出来的tf-idf值左边是词的id，右边是词的tfidf值

# 计算整个文档TF-IDF
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
      print(doc)

lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
documents = lsi_model[corpus]
query = [(0, 1), (1, 1), (2, 1)]
query_vec = lsi_model[query]
print(query_vec)

from gensim import similarities
index = similarities.MatrixSimilarity(documents)
# 可以通过save()和load()方法持久化这个相似度矩阵：
index.save('deerwester.index')
index = similarities.MatrixSimilarity.load('deerwester.index')
# 检查与所有语料中的余弦相似度
sims = index[query_vec]
print(sims)





# LDA主题模型
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
# 第一类主题，最具有代表性的前5个分词
print(lda.print_topic(1,topn=5))


# 设置最大主题数为50
hdp = models.hdpmodel.HdpModel(corpus, dictionary, T=50)
num_hdp_topics = len(hdp.print_topics())
print(num_hdp_topics)
