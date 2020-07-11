'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''


from skimage import io ,data
img = data.camera()
io.imshow(img)
io.show()

print(img)

# 用skimage中的io 读取图片
import skimage.io as io
import matplotlib.pyplot as plt
# 提取像素
imrgb = io.imread('Doraemon.jpg')
print('before reshape:',imrgb.shape)
# 第三个维度分别对应r,g,b像素值
# 将imrgb拼接成为一个行向量
imvec = imrgb.reshape(1,-1)
print('after reshape:',imvec.shape)
plt.gray() # 灰度图
fig, axes = plt.subplots(2,2,figsize=(12,10))
ax0,ax1,ax2,ax3= axes.ravel()
ax0.imshow(imrgb)
ax0.set_title('original image')
# Red通道
ax1.imshow(imrgb [:,:,0])
ax1.set_title('red channel')
# Green通道
ax2.imshow(imrgb [:,:,1])
ax2.set_title('green channel')
# Blue通道
ax3.imshow(imrgb [:,:,2])
ax3.set_title('blue channel')
plt.show()



import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.exposure import equalize_hist
def show_corners(corners, image):
    fig = plt.figure()
    # 灰度图
    plt.gray()
    plt.imshow(image)
    # 角点的像素当作坐标
    y_corner, x_corner = zip(*corners)
    plt.plot(x_corner, y_corner, 'or')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    plt.show()
imrgb = io.imread('Doraemon.jpg')
# 直方图均衡化，增强对比
imgray = equalize_hist(rgb2gray(imrgb))
corners = corner_peaks(corner_harris(imgray))
show_corners(corners, imrgb)


import matplotlib.pyplot as plt
from skimage import measure,data,color
#生成二值测试图像
imrgb = io.imread('Doraemon.jpg')
img=color.rgb2gray(imrgb)
#检测所有图形的轮廓
contours = measure.find_contours(img, 0.5)

#绘制轮廓
fig, axes = plt.subplots(1,2,figsize=(8,8))
ax0, ax1= axes.ravel()
ax0.imshow(img,plt.cm.gray)
ax0.set_title('original image')
rows,cols=img.shape
ax1.axis([0,rows,cols,0])
for n, contour in enumerate(contours):
    ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax1.axis('image')
ax1.set_title('outline')
plt.show()





