'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/16
'''

'''
Lasso回归和岭回归的实现
'''
import numpy as np
from sklearn import datasets, linear_model,  model_selection

boston = datasets.load_boston()
X_train, X_test, y_train, y_test = model_selection.train_test_split(boston.data, boston.target, test_size=0.2)


def model(model, X_train, X_test, y_train, y_test):
    # 进行训练
    model.fit(X_train, y_train)
    # 通过LinearRegression的coef_属性获得权重向量,intercept_获得b的值
    print("权重向量（斜率）:%s, 截距的值为:%.2f" % (model.coef_, model.intercept_))
    # 计算出损失函数的值
    print("回归模型的损失函数的值: %.2f" % np.mean((model.predict(X_test) - y_test) ** 2))
    # 计算预测性能得分
    print("预测性能得分: %.2f\n" % model.score(X_test, y_test))


if __name__ == '__main__':
    print(boston.feature_names)
    print("Losso回归开始训练: ")
    model(linear_model.Lasso(), X_train, X_test, y_train, y_test)
    print("Ridge回归开始训练: ")
    model(linear_model.Ridge(), X_train, X_test, y_train, y_test)
    print("ElasticNet回归开始训练: ")
    model(linear_model.ElasticNet(), X_train, X_test, y_train, y_test)
    print("线性回归开始训练: ")
    model(linear_model.LinearRegression(), X_train, X_test, y_train, y_test)
