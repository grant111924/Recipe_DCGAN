# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, time, itertools, imageio, pickle
from PIL import Image


def load_data(count,show=False):
    img=Image.open('test.png') 
    rsize=img.resize((64, 64))
    rsizeArr = np.asarray(rsize)
    imgplot = plt.imshow(rsizeArr) #piexl value = 0~255
    if show :plt.show()
    img=(rsizeArr-127.5)/127.5  #轉換到-1～1
    img4D=rsizeArr[np.newaxis,:,:,:]
    # 合併資料
    for i in range(count):
        if i == 0:
            imgData=img4D
        else :   
            imgData=np.concatenate((imgData,img4D),axis=0)
    print("Data generation finished..." )
    return imgData
class GAN():
    def __init__(self,data,z_dim,train_epoch,lr,batch_size):
        self.data=data
        self.z_dim=z_dim
        self.train_epoch=train_epoch
        self.lr=lr
        self.batch_size=batch_size
    def generator(x,isTrain=True):
        print("generator")
        with tf.variable_scope('generator', reuse=reuse):
            
            print(x)
            x = tf.layers.dense(x, units=4 * 4* 512,activation=tf.nn.tanh)
            x = tf.reshape(x, shape=[-1, 4, 4, 512])
            print(x)    
            # 1st hidden layer
            debias1=tf.get_variable("debias1", [512], initializer=tf.constant_initializer(0))
            deconv1 =  tf.layers.conv2d_transpose(x, 512, [3,3], strides=(1, 1), padding='same')
            deconv1=deconv1+debias1
            deconv1 = tf.layers.batch_normalization(inputs=deconv1,axis=-1,momentum=0.999,epsilon=1e-5,center=True,scale=True,training = isTrain)
            deconv1=tf.nn.relu(deconv1)
            print(deconv1)

            # 2nd hidden layer
            debias2=tf.get_variable("debias2", [256], initializer=tf.constant_initializer(0))
            deconv2 = tf.layers.conv2d_transpose(deconv1, 256, [3,3], strides=(2, 2), padding='same')
            deconv2=deconv2+debias2
            deconv2 = tf.layers.batch_normalization(inputs=deconv2,axis=-1,momentum=0.999,epsilon=1e-5,center=True,scale=True,training = isTrain)
            deconv2=tf.nn.relu(deconv2)
            print(deconv2)
            
            # 3rd hidden layer
            debias3=tf.get_variable("debias3", [128], initializer=tf.constant_initializer(0))
            deconv3 = tf.layers.conv2d_transpose(deconv2, 128, [3,3], strides=(2, 2), padding='same')
            deconv3=deconv3+debias3
            deconv3 = tf.layers.batch_normalization(inputs=deconv3,axis=-1,momentum=0.999,epsilon=1e-5,center=True,scale=True,training = isTrain)
            deconv3=tf.nn.relu(deconv3)
            print(deconv3)

            
            # 4th hidden layer
            debias4=tf.get_variable("debias4", [64], initializer=tf.constant_initializer(0))
            deconv4 = tf.layers.conv2d_transpose(deconv3, 64, [3,3], strides=(2, 2), padding='same')
            deconv4=deconv4+debias4
            deconv4 = tf.layers.batch_normalization(inputs=deconv4,axis=-1,momentum=0.999,epsilon=1e-5,center=True,scale=True,training = isTrain)
            deconv4=tf.nn.relu(deconv4)
            print(deconv4)
        

            # output layer
            debias5=tf.get_variable("debias5", [3], initializer=tf.constant_initializer(0))
            deconv5 = tf.layers.conv2d_transpose(deconv4, 3, [3,3], strides=(2, 2), padding='same')
            deconv5=deconv5+debias5
            deconv5 = tf.layers.batch_normalization(inputs=deconv5,axis=-1,momentum=0.999,epsilon=1e-5,center=True,scale=True,training = isTrain)
            o = tf.nn.tanh(deconv5)
            print(o)

            return o
    def discriminator(x,isTrain=True,reuse=False): 
        print("discriminator")
        with tf.variable_scope('discriminator', reuse=reuse):

            print(x)
            # 1st convolution + avg_pooling
            filter1 = tf.get_variable('weight1', [3, 3, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
            bias1 = tf.get_variable('bias1', [64], initializer=tf.constant_initializer(0))
            conv1 = tf.nn.conv2d(x, filter1, strides=[1, 2, 2, 1], padding='SAME')  
            print(conv1)
            conv1 = conv1 + bias1
            conv1 = tf.layers.batch_normalization(inputs=conv1,axis=-1,momentum=0.9,epsilon=1e-5,center=True,scale=True,training = isTrain)
            conv1 = tf.nn.leaky_relu(conv1)
            #conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(conv1)

            # 2nd convolution + avg_pooling
            filter2 = tf.get_variable('weight2', [3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
            bias2 = tf.get_variable('bias2', [128], initializer=tf.constant_initializer(0))
            conv2 =tf.nn.conv2d(conv1, filter2, strides=[1, 2, 2, 1], padding='SAME')
            print(conv2)
            conv2 = conv2 + bias2
            conv2 = tf.layers.batch_normalization(inputs=conv2,axis=-1,momentum=0.999,epsilon=1e-5,center=True,scale=True,training = isTrain)
            conv2 = tf.nn.leaky_relu(conv2)
            #conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(conv2)

            # 3rd convolution + avg_pooling
            filter3 = tf.get_variable('weight3', [3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
            bias3 = tf.get_variable('bias3',[256],initializer=tf.constant_initializer(0))
            conv3 =tf.nn.conv2d(conv2, filter3, strides=[1, 2, 2, 1], padding='SAME')
            print(conv3)
            conv3 = conv3 + bias3
            conv3 = tf.layers.batch_normalization(inputs=conv3,axis=-1,momentum=0.999,epsilon=1e-5,center=True,scale=True,training = isTrain)
            conv3 = tf.nn.leaky_relu(conv3)
            #conv3 = tf.nn.avg_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(conv3)


            # 4th convolution + avg_pooling
            filter4= tf.get_variable('weight4', [3, 3, 256, 512], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
            bias4 = tf.get_variable('bias4',[512],initializer=tf.constant_initializer(0)) 
            conv4 = tf.nn.conv2d(conv3, filter4,strides=[1, 2, 2, 1], padding='SAME') 
            print(conv4)
            conv4 = conv4 + bias4  
            conv4 = tf.layers.batch_normalization(inputs=conv4,axis=-1,momentum=0.999,epsilon=1e-5,center=True,scale=True,training = isTrain)
            conv4 = tf.nn.leaky_relu(conv4)
            #conv4 = tf.nn.avg_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(conv4)
            
            #full connected  layer 
            flatten = tf.reshape(conv4,(-1,4*4*512))
            logits=tf.layers.dense(inputs=flatten,units=1,activation=tf.nn.sigmoid)
            #o = tf.nn.sigmoid(logits)
            print(logits)
            
            return conv4,logits
    


if __name__ =="__main__":
    data=load_data(1024,False)

    