'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/19
'''
import sys
import numpy as np
import pandas as pd
df = pd.read_csv(sys.argv[1])
df = df.replace('?', np.nan)
df_feature = df.iloc[:, 0:-1]

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
features = imp.fit_transform(df_feature)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features1 = scaler.fit_transform(features)
data_list = []
for i in features1:
    temp = []
    for j in i:
        temp.append('%0.4f' % j)
    data_list.append(temp)
classes = df.iloc[:, -1].tolist()

# 读取配置文件
def conf_file(file):
    conf = pd.read_csv(file)
    # 将参数转换为列表
    parameters = conf.iloc[0].tolist()
    return parameters

from sklearn.preprocessing import LabelEncoder
labels = np.unique(classes)
lEnc = LabelEncoder()
lEnc.fit(labels)
label_encoder = lEnc.transform(classes)
numClass = len(labels)
label_encoder = label_encoder.astype(np.float64)

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
# 10-fold cross-validation
cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# KNN
from sklearn.neighbors import KNeighborsClassifier
def kNNClassifier(X, y, K):
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, np.asarray(X, dtype='float64'), y,
                             cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')

# 逻辑回归
from sklearn.linear_model import LogisticRegression
def logregClassifier(X, y):
    logreg = LogisticRegression(random_state=0)
    scores = cross_val_score(logreg, np.asarray(X, dtype='float64'), y,
                             cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')

# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
def nbClassifier(X, y):
    nb = GaussianNB()
    scores = cross_val_score(nb, np.asarray(X, dtype='float64'), y,
                             cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')

# 决策树
from sklearn.tree import DecisionTreeClassifier
def dtClassifier(X, y):
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    scores = cross_val_score(tree, np.asarray(X, dtype='float64'), y, cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')

# 集成学习Bagging
from sklearn.ensemble import BaggingClassifier
def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):
    bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=0),
                                n_estimators=n_estimators, max_samples=max_samples, random_state=0)
    scores = cross_val_score(bag_clf, np.asarray(X, dtype='float64'), y,
                             cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')

# 集成学习AdaBoost
from sklearn.ensemble import AdaBoostClassifier
def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=0),
                                 n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(ada_clf, np.asarray(X, dtype='float64'), y, cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')

# 集成学习Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
def gbClassifier(X, y, n_estimators, learning_rate):
    gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(gb_clf, np.asarray(X, dtype='float64'), y,
                             cv=cvKFold)
    print("{:.4f}".format(scores.mean()), end='')

# SVM
from sklearn.svm import SVC
def bestLinClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(kernel="linear", random_state=0), param_grid, cv=cvKFold, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_['C'])
    print(grid_search.best_params_['gamma'])
    print("{:.4f}".format(grid_search.best_score_))
    print("{:.4f}".format(grid_search.score(X_test, y_test)), end='')

# 随机森林
from sklearn.ensemble import RandomForestClassifier

def bestRFClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    param_grid = {'n_estimators': [10, 20, 50, 100], 'max_features': ['auto', 'sqrt', 'log2'],
                  'max_leaf_nodes': [10, 20, 30]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=0, criterion='entropy'), param_grid, cv=cvKFold,
                               return_train_score=True)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_['n_estimators'])
    print(grid_search.best_params_['max_features'])
    print(grid_search.best_params_['max_leaf_nodes'])
    print("{:.4f}".format(grid_search.best_score_))
    print("{:.4f}".format(grid_search.score(X_test, y_test)), end='')

def p():
    # 打印预处理数据
    for i in range(len(data_list)):
        for j in data_list[i]:
         print(j, end=',')
        if i < len(data_list) - 1:
            print(int(label_encoder[i]))
        else:
            print(int(label_encoder[i]), end='')

if sys.argv[2] == 'NN':
    parameter_list = conf_file(sys.argv[3])
    K = int(parameter_list[0])
    kNNClassifier(features1, label_encoder, K)

if sys.argv[2] == 'LR':
    logregClassifier(features1, label_encoder)

if sys.argv[2] == 'NB':
    nbClassifier(features1, label_encoder)
if sys.argv[2] == 'DT':
    dtClassifier(features1, label_encoder)

if sys.argv[2] == 'BAG':
    parameter_list = conf_file(sys.argv[3])
    n_estimators = int(parameter_list[0])
    max_samples = int(parameter_list[1])
    max_depth = int(parameter_list[2])
    bagDTClassifier(features1, label_encoder, n_estimators, max_samples, max_depth)

if sys.argv[2] == 'ADA':
    parameter_list = conf_file(sys.argv[3])
    n_estimators = int(parameter_list[0])
    learning_rate = parameter_list[1]
    max_depth = int(parameter_list[2])
    adaDTClassifier(features1, label_encoder, n_estimators, learning_rate, max_depth)

if sys.argv[2] == 'GB':
    parameter_list = conf_file(sys.argv[3])
    n_estimators = int(parameter_list[0])
    learning_rate = parameter_list[1]
    gbClassifier(features1, label_encoder, n_estimators, learning_rate)

if sys.argv[2] == 'RF':
    bestRFClassifier(features1, label_encoder)

if sys.argv[2] == 'SVM':
    bestLinClassifier(features1, label_encoder)

if sys.argv[2] == 'P':
    p()
