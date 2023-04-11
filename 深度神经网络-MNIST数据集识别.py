#引入必要的包
from keras.datasets import  mnist        #MNIST数据集
import cv2
from keras.models import Sequential                          #Sequential序贯模型
from keras.layers import Dense,Dropout,Activation           #全连接层  丢弃节点  激活函数
from keras.optimizers import  SGD                           #优化函数
import numpy as np
from keras.callbacks import TensorBoard

'''选择模型'''
model=Sequential()      #序贯模型


'''构建模型'''
model.add(Dense(500,input_shape=(784,)))    #第一个隐藏层  输出节点个数为500,输入节点个数为784
model.add(Activation('tanh'))     #指定tanh为激活函数
model.add(Dropout(0.5))           #每次丢弃掉一半节点的信息

model.add(Dense(500))             #第二个隐藏层
model.add(Activation('tanh'))     #指定tanh为激活函数
model.add(Dropout(0.5))           #每次丢弃掉一半节点的信息

model.add(Dense(500))             #第三个隐藏层
model.add(Activation('tanh'))     #指定tanh为激活函数
model.add(Dropout(0.5))           #每次丢弃掉一半节点的信息

model.add(Dense(10))             #输出层
model.add(Activation('softmax'))     #指定tanh为激活函数


'''训练设置和网络编译'''
sgd=SGD(lr=0.01,decay=1e-6)     #使用SGD为优化参数   初始化学习率（0.01）和学习率衰减值（1e-6）
model.compile(loss='categorical_crossentropy',optimizer=sgd)         #使用交叉熵作为loss函数
model.summary()      #查看网络结构


'''数据准备'''
(x_train,y_train),(x_test,y_test)=mnist.load_data()     #获取mnist数据集
print('原始训练样本的shape为:{}'.format(x_train.shape))
#将每个训练样本的输入变成一维
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])    #将每个训练样本的输入变成一维（由于mist的输入数据维度是(num，28，28)，这里需要把后面的维度直接拼起来）
print('训练样本的输入变成一维后的shape为:{}'.format(x_train.shape))
#将每个测试样本的输入变成一维
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])      #将每个测试样本的输入变成一维
print('*'*100)

#创建one-hot向量（将每个样本的预期输出变为一个One-Hot的10维向量，真实标签对应的位置设为1，其余设为0）
print('第一个训练输出值为:{}'.format(y_train[0]))
y_train=(np.arange(10)==y_train[:,None]).astype(int)        #对训练输出进行处理
print('对训练输出进行处理后的第一个训练输出值为:{}'.format(y_train[0]))
y_test=(np.arange(10)==y_test[:,None]).astype(int)        #对训练输出进行处理
print('*'*100)


#初始化Tensorboard对象
tb = TensorBoard('./logs')


'''网络训练'''
model.fit(x_train,y_train,batch_size=128,epochs=20,shuffle=True,verbose=2,validation_split=0.3,callbacks=[tb])
model.save('FC.model')    #保存模型
#加载网络模型
# model.load_weights('FC.model')

'''模型训练'''
scores=model.evaluate(x_test,y_test,batch_size=128,verbose=0)
print("The test loss is %f"%scores)


'''计算模型在测试集上的准确率'''
result=model.predict(x_test,batch_size=128,verbose=1)
print(result.shape)
print(result[0])
result_max=np.argmax(result,axis=1)               #得到网络预测的最大概率对应的类别序号
print("得到网络预测的最大概率对应的类别序号:%f" % result_max[0])

test_max=np.argmax(y_test,axis=1)                 #得到真实类别的最大概率对应的类别序号
result_bool=np.equal(result_max,test_max)         #得到预测值和真实值的样本
true_number=np.sum(result_bool)                    #正确结果的样本数
print("正确结果的样本数为:%f" % true_number)
print('The accuracy of the model is %f' % (true_number/len(result_bool)))       #验证结果的准确率


'''模型使用'''
test_image=cv2.imread("D:\desk\images\\img_4.png",0)
cv2.imshow("测试图片",test_image)
cv2.waitKey()
test_image=test_image.reshape(1,784)     #转换为（1，784）
result=model.predict(test_image,batch_size=1,verbose=0)    #预测
result_max=np.argmax(result,axis=1)     #这是结果的真实序号
print(result_max[0])       #打印结果
