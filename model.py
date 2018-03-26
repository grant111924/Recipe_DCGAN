# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import os, time, itertools, imageio, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def load_data():
    f=h5py.File("one.h5","r")
    origin_img=f['img']
    text=f['stvecs']
    origin_img=np.reshape(origin_img,(256,256,3))
    #print(origin_img)
    #print(plt.get_backend())
    #plt.imshow(origin_img)
    #label = 'Epoch'
    #plt.text(0.5, 0.04, label, ha='center')
    #plt.savefig("temp.png") # save figure
    #plt.show("temp.png")
    
    
        
    img=(origin_img-np.ones(origin_img.shape)*128)/256
    print(img)
    img4D=img[np.newaxis,:,:,:]
    for i in range(200):
        if i == 0:
            imgData=img4D
        else :
            imgData=np.concatenate((imgData,img4D),axis=0)
    print("Data generation finished..." )
    return imgData,text,origin_img
                
def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def generator(x, isTrain=True, reuse=False):
    print("generator")
    with tf.variable_scope('generator', reuse=reuse):
       
        # 1st hidden layer
        debias1=tf.get_variable("debias1", [128], initializer=tf.constant_initializer(0))
        deconv1 =  tf.layers.conv2d_transpose(x, 128, [3,3], strides=(1, 1), padding='same')
        deconv1=deconv1+debias1
        deconv1=tf.nn.relu(deconv1)
        print(deconv1)
         # 2nd hidden layer
        debias2=tf.get_variable("debias2", [64], initializer=tf.constant_initializer(0))
        deconv2 = tf.layers.conv2d_transpose(deconv1, 64, [3,3], strides=(2, 2), padding='same')
        deconv2=deconv2+debias2
        deconv2=tf.nn.relu(deconv2)
        print(deconv2)
        # 3rd hidden layer
        debias3=tf.get_variable("debias3", [32], initializer=tf.constant_initializer(0))
        deconv3 = tf.layers.conv2d_transpose(deconv2, 32, [3,3], strides=(2, 2), padding='same')
        deconv3=deconv3+debias3
        deconv3=tf.nn.relu(deconv3)
        print(deconv3)
        # output layer
        deconv4 = tf.layers.conv2d_transpose(deconv3, 3, [3,3], strides=(2, 2), padding='same')
        o = tf.nn.tanh(deconv4)
        print(deconv4)
        return o

def discriminator(x, isTrain=True, reuse=False):
    print("discriminator")
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st convolution + max_pooling
        filter1 = tf.get_variable('weight1', [3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias1 = tf.get_variable('bias1', [32], initializer=tf.constant_initializer(0))
        conv1 = tf.nn.conv2d(x, filter1, strides=[1, 1, 1, 1], padding='SAME')  
        print(conv1)
        conv1 = conv1 + bias1
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv1)

        # 2nd convolution + max_pooling
        filter2 = tf.get_variable('weight2', [3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias2 = tf.get_variable('bias2', [64], initializer=tf.constant_initializer(0))
        conv2 =tf.nn.conv2d(conv1, filter2, strides=[1, 1,1, 1], padding='SAME')
        print(conv2)
        conv2 = conv2 + bias2
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv2)

        # 3rd convolution + max_pooling
        filter3 = tf.get_variable('weight3', [3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias3 = tf.get_variable('bias3',[128],initializer=tf.constant_initializer(0))
        conv3 =tf.nn.conv2d(conv2, filter3, strides=[1, 1, 1, 1], padding='SAME')
        print(conv3)
        conv3 = conv3 + bias3
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.avg_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv3)

        # output layer
        filter4 = tf.get_variable('weight4', [3, 3, 128, 1], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        bias4 = tf.get_variable('bias4',[1],initializer=tf.constant_initializer(0)) 
        conv4 = tf.nn.conv2d(conv3, filter4, strides=[1, 1, 1, 1], padding='SAME') 
        conv4 = conv4 + bias4  
        o = tf.nn.sigmoid(conv4)
        print(conv4)
        return o, conv4
fixed_z_ = np.random.normal(0, 1, (1, 32, 32, 100))
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})
    newImg=np.reshape(test_images, (256, 256, 3))*256+np.ones((256,256,3))*128
    newImg=newImg.astype(np.uint8)
    plt.imshow(newImg)
    label = 'Epoch {0}'.format(num_epoch)
    plt.text(0.5, 0.04, label, ha='center')

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
train_epoch = 10000
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
    if epoch%100 == 0 and epoch != 0:
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
for e in range(train_epoch/100):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

sess.close()