'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/11
'''


import nltk
from nltk.text import Text
sentence = "Today's weather is good, very windy and sunny."
tokens = nltk.word_tokenize(sentence)
# 分词
print(tokens)
# 标记词性
tagged = nltk.pos_tag(tokens)
print(tagged)
# 查看对应单词的位置和个数
t = Text(tokens)
print(t.index('good'))
print(t.count('good'))

from nltk.corpus import brown
# brown语料的类别
print(brown.categories())
files = brown.fileids()
print(len(files))
print(len(brown.words()))
print(len(brown.sents()))
for w in brown.words(categories=['government','news']):
    print(w +' ',end='')
    if(w is '.'): # 一个句子换行
        print()

