import  cv2

img1=cv2.imread("D:\\desk\\images\\box.png")
img2=cv2.imread("D:\\desk\\images\\box_in_scene.png")

#SIFT特征检测（ 创建SIFT特征提取对象）
sift=cv2.SIFT_create(100)
#计算图像特征点
kp1,res1=sift.detectAndCompute(img1,None)     #kp为特征点列表，res为特征点的特征向量
kp2,res2=sift.detectAndCompute(img2,None)
#创建暴力匹配对象
bf=cv2.BFMatcher()

#用暴力匹配进行特征匹配
matches=bf.match(res1,res2)
# 匹配上的特征点会用线段进行连接
matches_img=cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)


# KNN匹配
matches_bf =bf.knnMatch(res1,res2,k=2)
# 匹配上的特征点会用线段进行连接
matches_img_bf =cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches_bf,None,flags=2)


good_matches=[]
#判断一个匹配是否可靠，保留可靠，去掉不可靠
for m in matches_bf:     #拿到img1中一个特征点的匹配
    #判断距离比例是否小于一个阈值
    if m[0].distance/m[1].distance<0.7:
        #把m[0]保存下来
        good_matches.append(m[0])

cv2.imshow("matches_img",matches_img)
cv2.imshow("matches_img_bf",matches_img_bf)

cv2.waitKey()
