'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/2
'''

import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn import preprocessing
from skimage import color

# 加载图像，并对数据进行规范化
def load_data(filePath):
    # 读文件
    f = open(filePath,'rb')
    data = []
    # 得到图像的像素值
    img = image.open(f)
    # 得到图像尺寸 = 512, 512
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 得到点(x,y)的三个通道值
            c1, c2, c3 = img.getpixel((x, y))
            data.append([c1, c2, c3])
    f.close()
    # 采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data), width, height


# 图片目录
pic_dir = './pic/'
# 步骤一：加载数据，得到规范化的结果imgData，以及图像尺寸
img, width, height = load_data(pic_dir+'Lena.png')

# 用K-Means对图像进行K聚类，这里的k是颜色的个数
n_clusters = 10

# max_iter 最大迭代数
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(img)
label = kmeans.predict(img)

# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])

# 聚类结果展示一:创建个新图像pic_mark，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark = image.new('L', (width, height))
for x in range(width):
    for y in range(height):
        # 根据类别设置图像灰度, 类别0 灰度值为255， 类别1 灰度值为127
        pic_mark.putpixel((x, y), int(256/(label[x][y]+1))-1)
pic_mark.save(pic_dir+'图像聚类的灰度值结果.jpg', 'JPEG')

# 聚类结果展示二：将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(label)*255).astype(np.uint8)
label_color = label_color.transpose(1,0,2)
images = image.fromarray(label_color)
images.save(pic_dir+'图像聚类的RGB结果.jpg')

# 聚类结果展示三：创建个新图像img，用来保存图像聚类压缩后的结果
img = image.new('RGB', (width, height))
for x in range(width):
    for y in range(height):
        c1 = kmeans.cluster_centers_[label[x, y], 0]
        c2 = kmeans.cluster_centers_[label[x, y], 1]
        c3 = kmeans.cluster_centers_[label[x, y], 2]
        img.putpixel((x, y), (int(c1*256)-1, int(c2*256)-1, int(c3*256)-1))
img.save(pic_dir+'图像聚类压缩结果.jpg')
