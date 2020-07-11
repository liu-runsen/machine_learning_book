'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/24
'''
import pandas as pd
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import LabelEncoder
data = pd.read_excel("数据.xlsx")
label_encoder=LabelEncoder()
integer_encoded = label_encoder.fit_transform(data.iloc[:35,-1].values)
X = data.iloc[:,0:3].values
data.iloc[:35,-1] = integer_encoded
print(integer_encoded)
# [1 1 1 2 2 2 3 3 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5 5 5 6 6 6 6 6 6 0 0 0 0]


data.iloc[35:,-1] =-1
y = data.iloc[:,-1].values
label_prop_model = LabelPropagation()
label_prop_model.fit(X, y)
clusterResult = label_prop_model.predict(X)
Result = list(label_encoder.inverse_transform(clusterResult))
print(Result[-10:])
# ['3C', '3C', '3C', '3C', '3C', '3C', '3C', '3C', '3B', '3B']
data = data.drop(columns=['类别'])
data['类别'] = Result
data.to_excel("结果.xlsx")