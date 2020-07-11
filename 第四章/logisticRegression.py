'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/16
'''

'''
Pima Indians数据集为糖尿病患者医疗记录数据，是一个典型二分类问题。该数据集最初来自国家糖尿病，消化肾脏疾病研究所。数据集的目标是基于数据集中包含的某些诊断测量来诊断性的预测：患者是否患有糖尿病。
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve

# 读取数据
data = pd.read_csv('diabetes.csv')
# Z-score标准化
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(data.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = data.Outcome

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

# 逻辑回归
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('逻辑回归在测试集中分类的精确率:{}\n'.format(logreg.score(X_test, y_test)))
# 混淆矩阵
confusion_matrix(y_test,y_pred)
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# 绘制ROC曲线
y_pred_proba = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Logistic Regression')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.show()

from sklearn.metrics import roc_auc_score
print("ROC Accuracy: {}".format(roc_auc_score(y_test,y_pred_proba)))
