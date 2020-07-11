'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/11
'''

from pyhanlp import *
# 分词和词性标注
sentence = "我爱自然语言处理技术！"
s_hanlp = HanLP.segment(sentence)
for term in s_hanlp:
print(term.word, term.nature)


# 依存句法分析
print(HanLP.parseDependency(sentence))

document = u'自然语言处理（natural language processing）是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。自然语言处理是一门融语言学、计算机科学、数学于一体的科学。'
doc_keyword = HanLP.extractKeyword(document, 3)
for word in doc_keyword:
    print(word)


# 短语提取
phraseList = HanLP.extractPhrase(document, 3)
print(phraseList)


# 摘要提取
doc_keysentence = HanLP.extractSummary(document, 3)
for key_sentence in doc_keysentence:
    print(key_sentence)

# 感知机词法分析器
PerceptronLexicalAnalyzer = JClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
analyzer = PerceptronLexicalAnalyzer()
print(analyzer.analyze("自然语言处理是一门融语言学、计算机科学、数学于一体的科学")))



# 中国人名识别
NER = HanLP.newSegment().enableNameRecognize(True)
print(NER.seg('范冰冰汪峰那英周杰伦王俊凯王源林俊杰迪丽热巴易烊千玺'))


# 音译人名识别
sentence = '微软的比尔盖茨、Facebook的扎克伯格跟桑德博格、亚马逊的贝索斯、苹果的库克，这些硅谷的科技人'
person_ner = HanLP.newSegment().enableTranslatedNameRecognize(True)
print(person_ner.seg(sentence))

Jianti = HanLP.convertToSimplifiedChinese("我愛自然語言處理技術！")
Fanti = HanLP.convertToTraditionalChinese("我爱自然语言处理技术！")
print(Jianti)
print(Fanti)


