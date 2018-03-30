#coding: utf-8
import random
import numpy as np
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
# from load_face_datasets import load_dataset, IMAGE_SIZE
from keras.callbacks import TensorBoard
from keras.models import load_model

#数据
class Dataset:
    def __init__(self):
        #训练集
        self.train_images = None
        self.train_labels = None
        #验证集
        self.valid_images = None
        self.valid_labels = None
        #测试集
        self.test_images = None
        self.test_labels = None
        #数据集加载路径
        # self.path_name = path_name
        #当前库采用的维度顺序
        self.input_shape = None

    #加载数据集并交叉验证
    def load(self,img_rows = 64, img_cols = 64,img_channels = 3, nb_classes = 50):
        #加载数据集到内存
        train_images = np.load("./train_images.npy")
        train_labels = np.load("./train_labels.npy")
        valid_images = np.load("./valid_images.npy")
        valid_labels = np.load("./valid_labels.npy")
        test_images = np.load("./test_images.npy")
        test_labels = np.load("./test_labels.npy")
        #30%作为验证，70%训练,从全部数据中随机选取数据建立训练集和验证集
        # train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
        #                                                                           random_state=random.randint(0, 100))
        # #测试集选择的比例为0.5`
        # _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
        #                                                   random_state=random.randint(0, 100))
        # 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        if K.image_dim_ordering() == 'tf':
            train_images = train_images.reshape(train_images.shape[0],img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows,img_cols,img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows,img_cols,img_channels)
            self.input_shape = (img_rows,img_cols,img_channels)
        # 输出训练集、验证集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')
        # 使用categorical_crossentropy作为损失函数,根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)
        # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')
        # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255
            valid_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels
#build CNN
class Model:
    def __init__(self):
        self.model = None
    #build model
    def build_model(self, images, nb_classes = 50):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32,kernel_size=(3, 3),strides=(1, 1),
                              padding='same',input_shape=dataset.input_shape))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Dropout(0.25))

        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        # self.model.summary()
        # plot_model(self.model, to_file='model1.png', show_shapes=True)
        # 训练模型
    def train(self, dataset, batch_size= 64, epochs = 1,data_augmentation = True):
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model.fit(dataset.train_images,dataset.train_labels,batch_size = batch_size,epochs = epochs,
                                   validation_data=(dataset.valid_images, dataset.valid_labels), shuffle = True,
                                                  callbacks=[TensorBoard(log_dir='./face')])

        print("Test--------------------------")
        # self.model.save('my_model')
        # del self.model
        # model = load_model('my_model')
        loss, accuracy = self.model.evaluate(dataset.test_images, dataset.test_labels)#batch_size 默认值３２
        print ('\n')
        print (' test loss: '), loss
        print (' test accuracy: '), accuracy

       # def Evaluate(self, dataset):
    #     score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
    #     print(" %s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

dataset = Dataset()
dataset.load()
model = Model()
model.build_model(dataset)
print("Train-------------------------")
model.train(dataset)

# print("Test--------------------------")
# model.Evaluate(dataset)



