'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/11
'''

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'D:\stanfordnlp', lang='zh')
sentence = '我是中国人，来自东莞'
# 分词
print(nlp.word_tokenize(sentence))
# 单词成分
print( nlp.pos_tag(sentence) )
# 命名实体识别
print(nlp.ner(sentence))
# 依赖关系分析
print(nlp.dependency_parse(sentence))
# 句法分析
print(nlp.parse(sentence))