'''
@Author ：毛利
'''

import os

# Pyspark配置
os.environ['PYSPARK_PYTHON'] = '/usr/local/python3/bin/python3'


from pyspark.sql import SparkSession
spark = SparkSession.builder.master('spark://node01:7077').appName('learn_ml').getOrCreate()

# 载入数据
df = spark.read.csv('hdfs://node01:9000/mushrooms.csv', header=True, inferSchema=True, encoding='utf-8')

# 先使用StringIndexer将字符转化为数值，然后将特征整合到一起
from pyspark.ml.feature import StringIndexer, VectorAssembler
old_columns_names = df.columns
print(old_columns_names)
new_columns_names = [name+'-new' for name in old_columns_names]
for i in range(len(old_columns_names)):
    indexer = StringIndexer(inputCol=old_columns_names[i], outputCol=new_columns_names[i])
    df = indexer.fit(df).transform(df)
vecAss = VectorAssembler(inputCols=new_columns_names[1:], outputCol='features')
df = vecAss.transform(df)
# 更换label列名
df = df.withColumnRenamed(new_columns_names[0], 'label')

# 创建新的只有label和features的表
data = df.select(['label', 'features'])

# 数据概观
print(data.show(5, truncate=0))


# 将数据集分为训练集和测试集
train_data, test_data = data.randomSplit([4.0, 1.0], 100)

from pyspark.ml.classification import LinearSVC
svm = LinearSVC()
svmModel = svm.fit(train_data)
result = svmModel.transform(test_data)

# accuracy
print(result.filter(result.label == result.prediction).count()/result.count())
# 0.9797172710510141
