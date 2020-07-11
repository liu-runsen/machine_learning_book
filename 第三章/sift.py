'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''



import cv2

def recognition(image_path,save_image):
# 调用sift算法
    sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.imread(image_path)
    # cv2.COLOR_BGR2GRAY灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(img, None)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
    # 保存图片
    img_save = cv2.drawKeypoints(img, kp, img, color=(255, 0, 255))
    cv2.imshow('point', img_save)
    cv2.imwrite('{}'.format(save_image), img)
    cv2.waitKey(0)

if __name__ == '__main__':
    recognition(image_path='iris.png',save_image='iris_save.png')
