'''（模板匹配）'''

import cv2, os
import numpy as np

# 读取模板
template_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉',
            '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
            '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']

# 图片预处理
def pre_processing(image):
    '''对图片进行预处理，保证后续操作的结果'''
    #第一步：将图片的高转换为120，保证各个函数传入参数的适用性
    target_height = 120
    h, w, _ = image.shape
    resize_ratio = target_height/h
    image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio)
    #第二部：去噪
    image = cv2.GaussianBlur(image, (3, 3), 0)
    #第三步：转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray_image", gray_image)
    # cv2.waitKey()
    return gray_image



# 图像分割
def get_segment(gray_image):
    '''得到图像分割结果，区分出字符和非字符区域'''
    #第一步:使用0tsu阅值分割算法对分割出字符和非字符区域
    ret,  thresh_image = cv2. threshold(gray_image, 200, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #第二步:比较分割结果中白色和黑色像素数量，保证字符区域是白色
    area_white = 0
    area_black = 0
    height, width = gray_image.shape
    for i in range(width):
        for j in range(height):
            if thresh_image[j, i]==0:
                area_black += 1
            else:
                area_white += 1
        if area_white > area_black:
            thresh_image = 255 - thresh_image
    # cv2.imshow("thresh_image", thresh_image)
    # cv2.waitKey()
    return thresh_image


# 单个字符外接矩形
def get_rectangles(thresh_image, draw_contours=False, draw_rectangle=False):
    '''获取一张阈值分割结果，返回单个字符的外接矩形坐标列表
        draw_contours:显示轮廓绘制的结果
        draw_rectangle:显示矩形绘制的结果
    '''
    # 轮廓检测
    th1 = cv2.dilate(thresh_image, None)  # 膨胀
    contours, hierachy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 轮廓可视化
    th1_bgr = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)  # 把单通道变成三通道，方便显示彩色
    if draw_contours:
        cv2.drawContours(th1_bgr, contours, -1, (0,0,255), 2)
        # cv2.imshow("contours", th1_bgr)
        # cv2.waitKey()

    '''外接矩形（包围框的提取于绘制）'''
    words = []  # 保存所有字符的外接矩形
    height, width = th1.shape  # 拿到整张图片的高和宽
    for contour in contours:  # 拿到每条轮廓的点的坐标
        rect = cv2.boundingRect(contour)  # 获取当前轮廓的外接矩形
        # 只保留高宽比在1.5-3.5范围内的矩形，并且这个矩形的高/整张图像的高大于0.3
        if rect[3] / rect[2] > 1.5 and rect[3] / rect[2] < 3.5 and rect[3] / height > 0.3:
            words.append(rect)  # 将当前矩形加入矩形列表
            if draw_rectangle:
                cv2.rectangle(th1_bgr, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                             (255, 255, 0), 3)  # 绘制矩形
                # cv2.imshow("rectangle", th1_bgr)
                # cv2.waitKey()
    return words

# 单字符图片列表
def get_single_words(image, words):
    '''对矩形从左至右排序，并且提取每一个矩形的ori; image：二值图片，words：矩形列表'''
    #根据每个元素的第一个值进行从小到大的排序
    words.sort(key=lambda x:x[0])
    #提取单个字符的ori，把单个字符的图片保存在words——imgs中
    words_imgs = []
    for rect in words:
        #取左上角xy坐标分别为(rect[0], rect[1]), 宽和高分别为rect[2]和rect[3]的ori
        current_img = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        words_imgs.append(current_img)
        cv2.imshow("current_img",current_img)
        cv2.waitKey()
    return words_imgs


def read_directory(directory_name):
    '''遍历文件夹，返回文件夹下所有文件的路径'''
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list

def get_templates():
    '''获取字符模板图片路径'''
    # 获取中文字符模板路径
    chinese_words = []
    for i in range(34, 64):
        c_word = read_directory('D:\\desk\\images\\refer1\\' + template_chars[i])
        chinese_words.append(c_word)
    # 获取英文和数字字符模板路径
    english_words = []
    for i in range(0, 34):
        c_word = read_directory('D:\\desk\\images\\refer1\\' + template_chars[i])
        english_words.append(c_word)
    return chinese_words, english_words

def get_charactor(char_img, templates, match_chinese=True):
    '''根据字符模板获取最佳匹配字符，并返回
    char_img: 要识别的字符图片， templates: 模板路径列表，natch_chinese: 是否匹配中文'''
    best_score = 0 #记录最高得分
    best_char = '' #记录最高得分对应的字符
    # TO DO：遍历所有字符模板，i代表字符序号，返回最高得分的字符
    for i in range(len(templates)):
        current_char_paths = templates[i] #拿到第i个字符所有模板图片路径
        for current_char_path in current_char_paths: #拿到当前的模板图片路径
            #读取模板图片
            templates_img = cv2.imdecode(np.fromfile(current_char_path, dtype=np.uint8), 1)
            #对模板图片进行阈值分割
            templates_img = cv2.cvtColor(templates_img, cv2.COLOR_BGR2GRAY)
            ret, templates_img = cv2.threshold(templates_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #将要识别的字符图片缩放到和模板图片一样的大小
            height, width = templates_img.shape
            char_img = cv2.resize(char_img, (width, height))
            #进行模板匹配
            result = cv2.matchTemplate(char_img, templates_img, cv2.TM_CCOEFF)
            #记录最高的得分和对应的字符
            if result[0][0] > best_score:
                best_score = result[0][0]
                if match_chinese:
                    best_char = template_chars[i+34]
                else:
                    best_char = template_chars[i]
    return best_char



rawImage = cv2.imread("D:\\desk\\images\\car_license\\test1.png")
cv2.imshow("rawImage",rawImage)
#预处理
gray_image = pre_processing(rawImage)
# 分割字符和非字符区域
thresh_image = get_segment(gray_image)
#得到单个字符的外接矩形坐标
words_rectangle = get_rectangles(thresh_image, draw_contours=True,draw_rectangle=True)
#得到单个字符图片的列表
words_imgs = get_single_words(thresh_image, words_rectangle)
# #读取模板
chinese_words, english_words = get_templates()
# #检测中文字符(第一个字符)
chinese_word_img = words_imgs[0]
best_word = get_charactor(chinese_word_img,  chinese_words,match_chinese=True)
print(best_word,end='  ')
#检测数字和英文字符
for  i in range(1,  len(words_imgs)):#得到第1个字符
    words_img = words_imgs[i]
    best_word = get_charactor (words_img, english_words, match_chinese=False)
    print(best_word,end='  ')
