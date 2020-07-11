'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/11
'''

import jieba
from jieba import posseg as pseg

print("jieba分词全模式：")
seg_list = jieba.cut("我是中国人，来自东莞", cut_all=True)
# 全模式
print("Full Mode: " + "/ ".join(seg_list))

print("jieba分词精确模式：")
seg_list = jieba.cut("我是中国人，来自东莞", cut_all=False)
# 精确模式
print("Default Mode: " + "/ ".join(seg_list))

print("jieba默认分词是精确模式：")
seg_list = jieba.cut("我是中国人，来自东莞")  # 默认是精确模式
print(", ".join(seg_list))

print("jiba搜索引擎模式：")
seg_list = jieba.cut_for_search("我是中国人，来自东莞")  # 搜索引擎模式
print(", ".join(seg_list))

strings="我是中国人，来自东莞"
words = pseg.cut(strings)
print("jieba词性标注：")
for word, flag in words:
    print('%s %s' % (word, flag))