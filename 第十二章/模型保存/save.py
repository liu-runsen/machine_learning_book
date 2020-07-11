'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/24
'''

from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X,y)
print(clf.predict(X[0:10]))
# [0 0 0 0 0 0 0 0 0 0]


import pickle #pickle模块

#保存Model(注:save文件夹要预先建立，否则会报错)
with open('clf.pickle', 'wb') as f:
    pickle.dump(clf, f)

#读取Model
with open('clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    #测试读取后的Model
    print(clf2.predict(X[0:10]))
# [0 0 0 0 0 0 0 0 0 0]



from sklearn.externals import joblib
joblib.dump(clf, 'clf.pkl')
#读取Model
clf3 = joblib.load('clf.pkl')
#测试读取后的Model
print(clf3.predict(X[0:10]))


from sklearn_porter import Porter
