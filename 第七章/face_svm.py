'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/4/28
'''



from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)






import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])
plt.show()



from time import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection  import GridSearchCV
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
t = time()
grid.fit(Xtrain, ytrain)
print('\n耗时：%f秒' % (time() - t))
print(grid.best_params_)
best_model = grid.best_estimator_
yfit = best_model.predict(Xtest)
print(yfit)

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    print(Xtest[i].shape)
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
plt.show()
from sklearn.metrics import classification_report,accuracy_score
print(accuracy_score(ytest,yfit))
print(classification_report(ytest, yfit,target_names=faces.target_names))

import seaborn as sns
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names,yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

