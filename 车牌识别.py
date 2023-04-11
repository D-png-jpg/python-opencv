import cv2

def get_single_words(image,words):
    '''对矩阵从左到右排序，并提取每个矩阵的roi     image:二值图片   words:矩阵列表'''
    words.sort(key=lambda x:x[0])
    #提取单个字符roi,把单个字符图片保存在words_imgs中
    words_imgs=[]
    for rest in words:
        #取左上角xy坐标为（rest[0],rest[1]）,宽和高为rest[2]  rest[3]的roi
        current_img=image[rest[1]:rest[1]+rest[3],rest[0]:rest[0]+rest[2]]
        words_imgs.append(words_imgs)
        cv2.imshow("current_img",current_img)
        cv2.waitKey()      #键盘输入再往下执行
    return  words_imgs


img=cv2.imread("D:\\desk\\images\\car_license\\test1.png")    #传入车牌图片路径

#去噪
image=cv2.GaussianBlur(img,(3,3),0)

#转为灰度图
gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Ostu阈值分割
ret, th1 = cv2.threshold(gray1, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

'''轮廓检测与绘制'''
#检测轮廓(外轮廓)
th1=cv2.dilate(th1,None)      #膨胀，保证同一个字符只有一个外轮廓
contours,hierarchy=cv2.findContours(th1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#轮廓可视化
th1_bgr=cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR)     #转为三通道图

# cv2.drawContours(th1_bgr,contours,-1,(0,0,255),2)    #轮廓可视化

'''包围框获取'''
words=[]         #保存包围框信息
height,width=th1.shape
for contour in contours:     #对于每一条轮廓
    rest=cv2.boundingRect(contour)       #得到这条轮廓的外接矩阵
    #只有高宽比在1.5到3.5之间，且高 度比图片高度大于0.3的矩阵才保留
    if rest[3]/rest[2]>1.5 and rest[3]/rest[2]<3.5 and rest[3]/height>0.3:
        words.append(rest)         #将当前矩形加入矩形列表
        cv2.rectangle(th1_bgr,(rest[0],rest[1]),(rest[0]+rest[2],rest[1]+rest[3]),(0,0,255),3)    #绘制矩形

        cv2.imshow("th1_bgr",th1_bgr)
        cv2.waitKey()     #键盘输入再往下执行

words_img=get_single_words(th1,words)
#显示
# cv2.imshow("img",img)
cv2.imshow("th1",th1)
# cv2.imshow("th1_bgr",th1_bgr)

cv2.waitKey()
