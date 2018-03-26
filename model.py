# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import os, time, itertools, imageio, pickle
def load_data():
    f=h5py.File("one.h5","r")
    origin_img=f['img']
    text=f['stvecs']
    img=np.reshape(origin_img,(256,256,3),np.float32)
    img=(img-np.ones(img.shape)*128)/256
    img4D=img[np.newaxis,:,:,:]
    for i in range(200):
        if i == 0:
            imgData=img4D
        else :
            imgData=np.concatenate((imgData,img4D),axis=0)
            #print(imgData.shape)
    print("Data generation finished..." )
    return imgData,text,origin_img
                
def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def generator(x, isTrain=True, reuse=False):
    print("generator")
    with tf.variable_scope('generator', reuse=reuse):
        print(x)
       
        # 1st hidden layer
        conv1 =  tf.layers.conv2d_transpose(x, 128, [4, 4], strides=(1, 1), padding='same')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
        print(conv1)
        """
        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        print(conv2)
        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        print(conv3)
        
        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        print(conv4)
        """
         # 5th hidden layer
        conv5 = tf.layers.conv2d_transpose(lrelu1, 64, [4,4], strides=(2, 2), padding='same')
        lrelu5 = lrelu(tf.layers.batch_normalization(conv5, training=isTrain), 0.2)
        print(conv5)
        # 6th hidden layer
        conv6 = tf.layers.conv2d_transpose(lrelu5, 32, [4,4], strides=(2, 2), padding='same')
        lrelu6 = lrelu(tf.layers.batch_normalization(conv6, training=isTrain), 0.2)
        print(conv6)
        # output layer
        conv7 = tf.layers.conv2d_transpose(lrelu6, 3, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv7)
        print(conv7)
        return o

def discriminator(x, isTrain=True, reuse=False):
    print("discriminator")
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st convolution + max_pooling
        filter1 = tf.get_variable('weight1', [4, 4, 3, 32], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias1 = tf.get_variable('bias1', [32], initializer=tf.constant_initializer(0))
        conv1 = tf.nn.conv2d(x, filter1, strides=[1, 2, 2, 1], padding='SAME')  
        print(conv1)
        conv1 = conv1 + bias1
        conv1 = tf.nn.relu(conv1)
        #conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv1)

        # 2nd convolution + max_pooling
        filter2 = tf.get_variable('weight2', [4, 4, 32, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias2 = tf.get_variable('bias2', [64], initializer=tf.constant_initializer(0))
        conv2 =tf.nn.conv2d(conv1, filter2, strides=[1, 2, 2, 1], padding='SAME')
        print(conv2)
        conv2 = conv2 + bias2
        conv2 = tf.nn.relu(conv2)
        #conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv2)

        # 3rd convolution + max_pooling
        filter3 = tf.get_variable('weight3', [4, 4, 64, 128], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias3 = tf.get_variable('bias3',[128],initializer=tf.constant_initializer(0))
        conv3 =tf.nn.conv2d(conv2, filter3, strides=[1, 2, 2, 1], padding='SAME')
        print(conv3)
        conv3 = conv3 + bias3
        conv3 = tf.nn.relu(conv3)
        #conv3 = tf.nn.avg_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv3)
        """
        # 4th convolution + max_pooling
        filter4 = tf.get_variable('weight4', [4, 4, 128, 256], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias4 = tf.get_variable('bias4',[256],initializer=tf.constant_initializer(0))
        conv4 = tf.nn.conv2d(conv3, filter4, strides=[1, 2, 2, 1], padding='SAME')  
        print(conv4) 
        conv4 = conv4 + bias4
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.nn.avg_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv4)
        
        # 5th hidden layer
        filter5 = tf.get_variable('weight5', [4, 4, 256, 512], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias5 = tf.get_variable('bias5',[512],initializer=tf.constant_initializer(0))
        conv5 = tf.nn.conv2d(conv4,filter5, strides=[1, 2, 2, 1], padding='SAME')   
        conv5 = conv5 + bias5
        conv5 = tf.nn.relu(conv5)
        conv5 = tf.nn.avg_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv5)

        # 6th hidden layer
        filter6 = tf.get_variable('weight6', [4, 4, 512, 1024], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias6 = tf.get_variable('bias6',[1024],initializer=tf.constant_initializer(0))  
        conv6 = tf.nn.conv2d(conv5, filter6, strides=[1, 2, 2, 1], padding='SAME')   
        conv6 = conv6 + bias6
        conv6 = tf.nn.relu(conv6)
        conv6 = tf.nn.avg_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv6)
       
        # output layer
        filter7 = tf.get_variable('weight7', [4, 4, 1024, 1], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias7 = tf.get_variable('bias7',[1],initializer=tf.constant_initializer(0)) 
        conv7 = tf.nn.conv2d(conv6, filter7, strides=[1, 1, 1, 1], padding='SAME') 
        conv7 = conv7 + bias7  
        o = tf.nn.sigmoid(conv7)
        print(conv7)
        """
        # output layer
        filter4 = tf.get_variable('weight4', [4, 4, 128, 1], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias4 = tf.get_variable('bias4',[1],initializer=tf.constant_initializer(0)) 
        conv4 = tf.nn.conv2d(conv3, filter4, strides=[1, 1, 1, 1], padding='SAME') 
        conv4 = conv4 + bias4  
        o = tf.nn.sigmoid(conv4)
        print(conv4)
        return o, conv4

fixed_z_ = np.random.normal(0, 1, (25, 32, 32, 100))
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})
 
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        newImg=np.reshape(test_images[k], (256, 256, 3))*256+np.ones((256,256,3))*128
        newImg=newImg.astype(np.uint8)
        ax[i, j].imshow(newImg, cmap=None)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# training parameters
batch_size = 100
lr = 0.00001
train_epoch = 1000
#load recipe data
imgData,text,origin_img=load_data()

# variables : input
x = tf.placeholder(tf.float32, shape=(None, 256, 256, 3))
z = tf.placeholder(tf.float32, shape=(None, 32, 32, 100))
isTrain = tf.placeholder(dtype=tf.bool)


# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 32, 32, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 32, 32, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 32, 32, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# results save folder
root = 'DCGAN_results/'
model = 'DCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range( len(imgData) // batch_size):
        # update discriminator
        x_ = imgData[iter*batch_size:(iter+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 32, 32, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 32, 32, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTrain: True})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    if epoch%100 == 0:
        show_result((epoch + 100), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

sess.close()