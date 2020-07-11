'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/24
'''
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
pipeline = PMMLPipeline([
	("pca", PCA(n_components = 3)),
	("classifier", SVC())])
iris = load_iris()
pipeline.fit(iris.data, iris.target)
sklearn2pmml(pipeline, "iris_SVC.pmml", with_repr = True)
