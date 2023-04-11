import  cv2
import numpy as np

img=cv2.imread("D:\\desk\\images\\chessboard.jpg")
print(img.shape)
#转换为float32  单灰度通道
gary=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gary=np.float32(gary)


#使用Harris算法检测角点
dst=cv2.cornerHarris(gary,2,3,0.04)   #第二个参数为以角点为中心的方框大小  第三个参数Sobel算法中使用的卷积核大小

#将角点图片做膨胀处理，使角点明显
dst=cv2.dilate(dst,None)                 #第二个参数为kernel：结构元/卷积核，如果输入None则使用默认的3*3的矩阵
#将高于最大值0.01倍的结果图认为是角点，红色赋值
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow("img",img)
cv2.waitKey()
