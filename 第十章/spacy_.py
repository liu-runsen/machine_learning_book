'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/11
'''

import spacy
nlp = spacy.load('en')
doc = nlp('Weather is good, very windy and sunny.')
# 分词
for token in doc:
    print(token)
# 分句
for sent in doc.sents:
    print(sent)
# 词性
for token in doc:
    print('{}-{}'.format(token, token.pos_))
# 命名体识别
for ent in doc.ents:
    print('{}-{}'.format(ent, ent.label_))