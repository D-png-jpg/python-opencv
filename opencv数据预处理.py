'''
数据预处理：旋转（角度随机）+缩小      翻转（垂直）    亮度+对比度    清晰度（模糊）     几何变换+像素变换
'''
import os        #文件操作库
import random     #生成随机数的库
import cv2       #图像处理库
import numpy as np   #数组操作库
import shutil

'''修改文件名'''
def refilename(dirpath,dirpath02):       #dirpath为所在的文件夹名
    GOOD_dirpath=os.path.join(dirpath,"GOOD")
    NG_dirpath=os.path.join(dirpath,"NG")
    abnormal_dirpath=os.path.join(dirpath02,"abnormal")   #引脚缺失
    num=1      #图片名序号
    for img_name in os.listdir(GOOD_dirpath):           #os.listdir列出路径下所有文件名
        src=os.path.join(GOOD_dirpath,img_name)        #src：要修改的目录名
        dst=os.path.join(GOOD_dirpath,"good_"+str(num)+'.jpg')         #dst：修改后的目录名
        num+=1
        os.rename(src,dst)             #修改
    num=1
    for img_name in os.listdir(NG_dirpath):           #os.listdir列出路径下所有文件名
        src=os.path.join(NG_dirpath,img_name)        #src：要修改的目录名
        dst=os.path.join(NG_dirpath,"ng_"+str(num)+'.jpg')         #dst：修改后的目录名
        num+=1
        os.rename(src,dst)             #修改
    num=1      #图片名序号
    for img_name in os.listdir(abnormal_dirpath):           #os.listdir列出路径下所有文件名
        src=os.path.join(abnormal_dirpath,img_name)        #src：要修改的目录名
        dst=os.path.join(abnormal_dirpath,"abnormal_"+str(num)+'.jpg')         #dst：修改后的目录名
        num+=1
        os.rename(src,dst)             #修改
    return dst

'''数据预处理——旋转+缩小'''
def revolve1(dirpath,dirpath02):
    GOOD_dirpath=os.path.join(dirpath,"GOOD")
    NG_dirpath=os.path.join(dirpath,"NG")
    abnormal_dirpath=os.path.join(dirpath02,"abnormal")   #引脚缺失
    num=len(os.listdir(new_good))+1         #图片名序号
    for img_file in os.listdir(GOOD_dirpath):           #os.listdir列出路径下所有文件名
        # for angle in np.arange(15,360,30):           #旋转角度     在[15,360)中每个30取一个数
        for i in range(1,5):              #每张图片旋转5次
            angle = random.randint(1,360)   #在1到360度中随机选角度
            src=os.path.join(GOOD_dirpath,img_file)        #src：路径
            img = cv2.imread(src)                     # 以彩色模式读取图像文件
            h, w, _ = img.shape                       # 获取图像形状
            M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 0.4)        #定义旋转中心   (参数： 旋转中心，旋转角，缩放系数)
            new_img = cv2.warpAffine(img, M, (h, w))          #旋转
            cv2.imwrite(new_good+'/'+"good_"+str(num)+'.jpg',new_img)        #保存图片   （参数：保存路径，要保存的图像，编码格式）
            num+=1
            # if angle>=360:
            #     continue
    num=len(os.listdir(new_ng))+1        #图片名序号
    for img_file in os.listdir(NG_dirpath):  # os.listdir列出路径下所有文件名
        # for angle in np.arange(15, 360, 30):  # 旋转角度     在[15,360)中每个15取一个数
        for i in range(1,5):              #每张图片旋转15次
            angle = random.randint(1,360)   #在1到360度中随机选角度
            src = os.path.join(NG_dirpath, img_file)  # src：路径
            img = cv2.imread(src)  # 以彩色模式读取图像文件
            h, w, _ = img.shape  # 获取图像形状
            M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 0.4)  # 定义旋转中心   (参数： 旋转中心，旋转角，缩放系数)
            new_img = cv2.warpAffine(img, M, (h, w))  # 旋转
            cv2.imwrite(new_ng + '/' + "ng_" + str(num) + '.jpg', new_img)   #保存图片   （参数：保存路径，要保存的图像，编码格式）
            num += 1
            # if angle >= 360:
            #     continue
    num=len(os.listdir(new_abnormal))+1        #图片名序号
    for img_file in os.listdir(abnormal_dirpath):  # os.listdir列出路径下所有文件名
        # for angle in np.arange(15, 360, 30):  # 旋转角度     在[15,360)中每个15取一个数
        for i in range(1,3):              #每张图片旋转15次
            angle = random.randint(1,360)   #在1到360度中随机选角度
            src = os.path.join(abnormal_dirpath, img_file)  # src：路径
            img = cv2.imread(src)  # 以彩色模式读取图像文件
            h, w, _ = img.shape  # 获取图像形状
            M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 0.35)  # 定义旋转中心   (参数： 旋转中心，旋转角，缩放系数)
            new_img = cv2.warpAffine(img, M, (h, w))  # 旋转
            cv2.imwrite(new_abnormal + '/' + "abnormal_" + str(num) + '.jpg', new_img)   #保存图片   （参数：保存路径，要保存的图像，编码格式）
            num += 1

'''几何变换+像素变换'''
def revolve2(dirpath,dirpath02):
    GOOD_dirpath=os.path.join(dirpath,"GOOD")
    NG_dirpath=os.path.join(dirpath,"NG")
    abnormal_dirpath=os.path.join(dirpath02,"abnormal")   #引脚缺失
    num=len(os.listdir(new_good))+1         #图片名序号
    for img_file in os.listdir(GOOD_dirpath):           #os.listdir列出路径下所有文件名
        # for angle in np.arange(15,360,30):           #旋转角度     在[15,360)中每个30取一个数
        for i in range(1,5):              #每张图片旋转15次
            angle = random.randint(1,360)   #a在1到360度中随机选角度
            src=os.path.join(GOOD_dirpath,img_file)        #src：路径
            img = cv2.imread(src)                     # 以彩色模式读取图像文件
            h, w, _ = img.shape                       # 获取图像形状
            M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 0.35)        #定义旋转中心   (参数： 旋转中心，旋转角，缩放系数)
            new_img1 = cv2.warpAffine(img, M, (h, w))          #旋转
            # new_img = cv2.GaussianBlur(new_img1, (11, 11), 0)  #高斯模糊 参数0表示标准差取0
            liangdu=8
            h2,w2,k=new_img1.shape
            c = random.uniform(0.2, 1.8)  # 随机生成一个实数    对比度
            blank = np.zeros([h2, w2, k], img.dtype)  # 定义一张空白图像  np.zeros:用0填充的数组  （参数：形状，数据类型）
            new_img = cv2.addWeighted(new_img1, c, blank, 1 - c,liangdu)  # 权重加法函数   (参数：第一和第三个参数为需要融合相加的两副大小和通道数相等的图像，2、4为对应的权重，修正系数)
            ### 权重加法:src1 * alpha + src2 * beta + gamma ###
            cv2.imwrite(new_good+'/'+"good_"+str(num)+'.jpg',new_img)        #保存图片   （参数：保存路径，要保存的图像，编码格式）
            num+=1
            # if angle>=360:
            #     continue
    num=len(os.listdir(new_ng))+1        #图片名序号
    for img_file in os.listdir(NG_dirpath):  # os.listdir列出路径下所有文件名
        # for angle in np.arange(15, 360, 30):  # 旋转角度     在[15,360)中每个15取一个数
        for i in range(1,5):              #每张图片旋转10次
            angle = random.randint(1,360)   #在1到360度中随机选角度
            src = os.path.join(NG_dirpath, img_file)  # src：路径
            img = cv2.imread(src)  # 以彩色模式读取图像文件
            h, w, _ = img.shape  # 获取图像形状
            M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 0.4)  # 定义旋转中心   (参数： 旋转中心，旋转角，缩放系数)
            new_img1 = cv2.warpAffine(img, M, (h, w))  # 旋转
            # new_img = cv2.GaussianBlur(new_img1, (11, 11), 0)  #高斯模糊 参数0表示标准差取0
            liangdu=8
            h2,w2,k=new_img1.shape
            c = random.uniform(0.2, 1.8)  # 随机生成一个实数    对比度
            blank = np.zeros([h2, w2, k], img.dtype)  # 定义一张空白图像  np.zeros:用0填充的数组  （参数：形状，数据类型）
            new_img = cv2.addWeighted(new_img1, c, blank, 1 - c,liangdu)  # 权重加法函数   (参数：第一和第三个参数为需要融合相加的两副大小和通道数相等的图像，2、4为对应的权重，修正系数)
            ### 权重加法:src1 * alpha + src2 * beta + gamma ###
            cv2.imwrite(new_ng + '/' + "ng_" + str(num) + '.jpg', new_img)   #保存图片   （参数：保存路径，要保存的图像，编码格式）
            num += 1
            # if angle >= 360:
            #     continue
    num=len(os.listdir(new_abnormal))+1        #图片名序号
    for img_file in os.listdir(abnormal_dirpath):  # os.listdir列出路径下所有文件名
        # for angle in np.arange(15, 360, 30):  # 旋转角度     在[15,360)中每个15取一个数
        for i in range(1,3):              #每张图片旋转15次
            angle = random.randint(1,360)   #在1到360度中随机选角度
            src = os.path.join(abnormal_dirpath, img_file)  # src：路径
            img = cv2.imread(src)  # 以彩色模式读取图像文件
            h, w, _ = img.shape  # 获取图像形状
            M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 0.4)  # 定义旋转中心   (参数： 旋转中心，旋转角，缩放系数)
            new_img1 = cv2.warpAffine(img, M, (h, w))  # 旋转
            # new_img = cv2.GaussianBlur(new_img1, (11, 11), 0)  #高斯模糊 参数0表示标准差取0
            liangdu=8
            h2,w2,k=new_img1.shape
            c = random.uniform(0.2, 1.8)  # 随机生成一个实数    对比度
            blank = np.zeros([h2, w2, k], img.dtype)  # 定义一张空白图像  np.zeros:用0填充的数组  （参数：形状，数据类型）
            new_img = cv2.addWeighted(new_img1, c, blank, 1 - c,liangdu)  # 权重加法函数   (参数：第一和第三个参数为需要融合相加的两副大小和通道数相等的图像，2、4为对应的权重，修正系数)
            ### 权重加法:src1 * alpha + src2 * beta + gamma ###
            cv2.imwrite(new_abnormal + '/' + "abnormal_" + str(num) + '.jpg', new_img)   #保存图片   （参数：保存路径，要保存的图像，编码格式）
            num += 1

'''数据预处理——翻转（垂直）'''
def turn(dirpath,dirpath02):
    GOOD_dirpath = os.path.join(dirpath, "GOOD")
    NG_dirpath = os.path.join(dirpath, "NG")
    abnormal_dirpath=os.path.join(dirpath02,"abnormal")   #引脚缺失
    num=len(os.listdir(new_good))+1         #图片名序号
    for img_file in os.listdir(GOOD_dirpath):  # os.listdir列出路径下所有文件名
        src = os.path.join(GOOD_dirpath, img_file)  # src：路径
        img = cv2.imread(src)  # 以彩色模式读取图像文件
        new_img = cv2.flip(img, 0)  # 垂直翻转
        # new_img = cv2.flip(img, 1)  # 水平翻转
        # new_img = cv2.flip(img, -1)  # 水平和垂直翻转
        cv2.imwrite(new_good + '/' + "good_" + str(num) + '.jpg', new_img)     #保存图片   （参数：保存路径，要保存的图像，编码格式）
        num += 1
    num=len(os.listdir(new_ng))+1        #图片名序号
    for img_file in os.listdir(NG_dirpath):       #os.listdir列出路径下所有文件名
        src = os.path.join(NG_dirpath, img_file)  #src：路径
        img = cv2.imread(src)                       #以彩色模式读取图像文件
        new_img = cv2.flip(img, 0)                  #垂直翻转
        # new_img = cv2.flip(img, 1)                #水平翻转
        # new_img = cv2.flip(img, -1)               #水平和垂直翻转
        cv2.imwrite(new_ng + '/' + "ng_" + str(num) + '.jpg', new_img)     #保存图片   （参数：保存路径，要保存的图像，编码格式）
        num += 1
    num=len(os.listdir(new_abnormal))+1        #图片名序号
    for img_file in os.listdir(abnormal_dirpath):       #os.listdir列出路径下所有文件名
        src = os.path.join(abnormal_dirpath, img_file)  #src：路径
        img = cv2.imread(src)                       #以彩色模式读取图像文件
        new_img = cv2.flip(img, 0)                  #垂直翻转
        # new_img = cv2.flip(img, 1)                #水平翻转
        # new_img = cv2.flip(img, -1)               #水平和垂直翻转
        cv2.imwrite(new_abnormal + '/' + "abnormal_" + str(num) + '.jpg', new_img)     #保存图片   （参数：保存路径，要保存的图像，编码格式）
        num += 1

'''数据预处理——调整亮度+对比度'''
def contrast(dirpath,dirpath02,liangdu):
    GOOD_dirpath = os.path.join(dirpath, "GOOD")
    NG_dirpath = os.path.join(dirpath, "NG")
    abnormal_dirpath=os.path.join(dirpath02,"abnormal")   #引脚缺失
    num=len(os.listdir(new_good))+1         #图片名序号
    for img_file in os.listdir(GOOD_dirpath):  # os.listdir列出路径下所有文件名
        src = os.path.join(GOOD_dirpath, img_file)  # src：路径
        img = cv2.imread(src)  # 以彩色模式读取图像文件
        h, w, k = img.shape  # 获取图像形状
        c = random.uniform(0.2, 1.8)  # 随机生成一个实数    对比度
        blank = np.zeros([h, w,k ], img.dtype)    #定义一张空白图像  np.zeros:用0填充的数组  （参数：形状，数据类型）
        new_img = cv2.addWeighted(img, c, blank,1-c, liangdu)     #权重加法函数   (参数：第一和第三个参数为需要融合相加的两副大小和通道数相等的图像，2、4为对应的权重，修正系数)
        ### 权重加法:src1 * alpha + src2 * beta + gamma ###
        cv2.imwrite(new_good + '/' + "good_" + str(num) + '.jpg', new_img)  # 保存图片   （参数：保存路径，要保存的图像，编码格式）
        num += 1
    num=len(os.listdir(new_ng))+1        #图片名序号
    for img_file in os.listdir(NG_dirpath):  # os.listdir列出路径下所有文件名
        src = os.path.join(NG_dirpath, img_file)  # src：路径
        img = cv2.imread(src)  # 以彩色模式读取图像文件
        h, w, k = img.shape  # 获取图像形状
        c = random.uniform(0.2, 1.8)  # 随机生成一个实数    对比度
        blank = np.zeros([h, w,k ], img.dtype)    #定义一张空白图像  np.zeros:用0填充的数组  （参数：形状，数据类型）
        new_img = cv2.addWeighted(img, c, blank,1-c, liangdu)     #权重加法函数   (参数：第一和第三个参数为需要融合相加的两副大小和通道数相等的图像，2、4为对应的权重，修正系数)
        ### 权重加法:src1 * alpha + src2 * beta + gamma ###
        cv2.imwrite(new_ng + '/' + "ng_" + str(num) + '.jpg', new_img)  # 保存图片   （参数：保存路径，要保存的图像，编码格式）
        num += 1
    num = len(os.listdir(new_abnormal)) + 1  # 图片名序号
    for img_file in os.listdir(abnormal_dirpath):  # os.listdir列出路径下所有文件名
        src = os.path.join(abnormal_dirpath, img_file)  # src：路径
        img = cv2.imread(src)  # 以彩色模式读取图像文件
        h, w, k = img.shape  # 获取图像形状
        b = 10  # 亮度
        c = random.uniform(0.2, 1.8)  # 随机生成一个实数    对比度
        blank = np.zeros([h, w, k], img.dtype)  # 定义一张空白图像  np.zeros:用0填充的数组  （参数：形状，数据类型）
        new_img = cv2.addWeighted(img, c, blank, 1 - c, liangdu)  # 权重加法函数   (参数：第一和第三个参数为需要融合相加的两副大小和通道数相等的图像，2、4为对应的权重，修正系数)
        ### 权重加法:src1 * alpha + src2 * beta + gamma ###
        cv2.imwrite(new_abnormal + '/' + "abnormal_" + str(num) + '.jpg', new_img)  # 保存图片   （参数：保存路径，要保存的图像，编码格式）
        num += 1

'''数据预处理——调整清晰度（模糊）'''
def definition(dirpath,dirpath02):
    GOOD_dirpath = os.path.join(dirpath, "GOOD")
    NG_dirpath = os.path.join(dirpath, "NG")
    abnormal_dirpath=os.path.join(dirpath02,"abnormal")   #引脚缺失
    num=len(os.listdir(new_good))+1         #图片名序号
    for img_file in os.listdir(GOOD_dirpath):  # os.listdir列出路径下所有文件名
        src = os.path.join(GOOD_dirpath, img_file)  # src：路径
        img = cv2.imread(src)  # 以彩色模式读取图像文件
        # # 均值模糊
        # new_img = cv2.blur(img, (5, 5))
        # # 中值模糊
        # new_img = cv2.medianBlur(img, 5)
        # 高斯滤波,卷积核大小为5*5
        new_img = cv2.GaussianBlur(img, (11, 11), 0)    #参数0表示标准差取0
        cv2.imwrite(new_good + '/' + "good_" + str(num) + '.jpg', new_img)  # 保存图片   （参数：保存路径，要保存的图像，编码格式）
        num += 1
    num=len(os.listdir(new_ng))+1        #图片名序号
    for img_file in os.listdir(NG_dirpath):  # os.listdir列出路径下所有文件名
        src = os.path.join(NG_dirpath, img_file)  # src：路径
        img = cv2.imread(src)  # 以彩色模式读取图像文件
        # # 均值模糊
        # new_img = cv2.blur(img, (5, 5))
        # # 中值模糊
        # new_img = cv2.medianBlur(img, 5)
        # 高斯滤波,卷积核大小为5*5
        new_img = cv2.GaussianBlur(img, (11, 11), 0)    #参数0表示标准差取0
        cv2.imwrite(new_ng + '/' + "ng_" + str(num) + '.jpg', new_img)  # 保存图片   （参数：保存路径，要保存的图像，编码格式）
        num += 1
    num=len(os.listdir(new_abnormal))+1        #图片名序号
    for img_file in os.listdir(abnormal_dirpath):  # os.listdir列出路径下所有文件名
        src = os.path.join(abnormal_dirpath, img_file)  # src：路径
        img = cv2.imread(src)  # 以彩色模式读取图像文件
        new_img = cv2.GaussianBlur(img, (11, 11), 0)    #参数0表示标准差取0
        cv2.imwrite(new_abnormal + '/' + "abnormal_" + str(num) + '.jpg', new_img)  # 保存图片   （参数：保存路径，要保存的图像，编码格式）
        num += 1

'''复制图片(无划痕和引脚正常数据集相同)'''
def copy(dirpath,dirpath02):
    # 如果已存在相关目录，则删除
    if os.path.exists(dirpath02+'/normal'):     #判断是否存在
        shutil.rmtree(dirpath02+'/normal')        #删除目录
    GOOD_dirpath = os.path.join(dirpath, "GOOD")
    normal_dir = os.path.join(dirpath02, 'normal')     # 定义存储路径
    os.mkdir(normal_dir)                               # 根据路径创建训练集文件夹
    for img_file in os.listdir(GOOD_dirpath):  # os.listdir列出路径下所有文件名
        # 定义该图片文件的源路径
        src = os.path.join(GOOD_dirpath, img_file)
        # 定义图片文件保存的目标路径
        dst = os.path.join(normal_dir, img_file)
        # 将该图片文件从源路径复制到目标路径
        shutil.copyfile(src, dst)        #把无划痕的图片（增强前）复制到引脚正常
    for img_file in os.listdir(new_good):  # os.listdir列出路径下所有文件名
        # 定义该图片文件的源路径
        src = os.path.join(new_good, img_file)
        # 定义图片文件保存的目标路径
        dst = os.path.join(new_normal, img_file)
        # 将该图片文件从源路径复制到目标路径
        shutil.copyfile(src, dst)        #把无划痕的图片（增强后）复制到引脚正常

if __name__=="__main__":
    dirpath=r'F:/fenlei'      #分类文件夹路径
    dirpath02=r'F:/jiance'      #检测文件夹路径

    # 处理后的数据另放一个文件夹       如果已存在相关目录，则删除
    if os.path.exists(dirpath+'/new_good'):     #判断是否存在
        shutil.rmtree(dirpath+'/new_good')        #删除目录
    os.mkdir(dirpath+'/new_good')        #创建文件夹
    if os.path.exists(dirpath+'/new_ng'):     #判断是否存在
        shutil.rmtree(dirpath+'/new_ng')        #删除目录
    os.mkdir(dirpath+'/new_ng')        #创建文件夹
    if os.path.exists(dirpath02+'/new_abnormal'):     #判断是否存在
        shutil.rmtree(dirpath02+'/new_abnormal')        #删除目录
    os.mkdir(dirpath02+'/new_abnormal')        #创建文件夹
    if os.path.exists(dirpath02+'/new_normal'):     #判断是否存在
        shutil.rmtree(dirpath02+'/new_normal')        #删除目录
    os.mkdir(dirpath02+'/new_normal')        #创建文件夹
    new_good=dirpath+'/new_good'
    new_ng=dirpath+'/new_ng'
    new_abnormal=dirpath02+'/new_abnormal'
    new_normal=dirpath02+'/new_normal'

    liangdu=8   #亮度
    refilename(dirpath,dirpath02)    #修改文件名
    revolve1(dirpath,dirpath02)     #旋转+缩小
    revolve2(dirpath,dirpath02)     #几何变换+像素变换
    turn(dirpath,dirpath02)    #翻转
    contrast(dirpath,dirpath02,liangdu)   #亮度+对比度
    definition(dirpath,dirpath02)    #清晰度（模糊）
    copy(dirpath,dirpath02)        #复制图片(无划痕和引脚正常数据集相同)
