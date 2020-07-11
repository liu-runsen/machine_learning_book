import os

# Pyspark配置
os.environ['PYSPARK_PYTHON'] = '/usr/local/python3/bin/python3'
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# 初始化SparkSession和SparkContext
spark = SparkSession.builder.master("spark://node01:7077").appName("Housing prices").getOrCreate()
# 读取house-prices.csv
df = spark.read.csv("hdfs://node01:9000/house-prices.csv", header=True)
# 文本格式转化double数值格式
data = df.select(df.Price.cast('double'), df.SqFt.cast('double'), df.Bedrooms.cast('double'),
                 df.Bathrooms.cast('double'), df.Offers.cast('double'))

# 使用Sqrt房屋面积，Bedrooms卧式个数，Bathrooms浴室个数和Offers作为特征向量

assembler = VectorAssembler(inputCols=["SqFt", "Bedrooms", 'Bathrooms', 'Offers'], outputCol='features')
output = assembler.transform(data)
label_features = output.select("features", "Price").toDF('features', 'label')
label_features.show(truncate=False)

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

lrModel = lr.fit(label_features)

print("Coefficients: %s" % str(lrModel.coefficients))

print("Intercept: %s" % str(lrModel.intercept))

trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
