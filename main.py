# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:41:59 2018

@author: menguan
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

file = "D:/DATA/MNIST"

mnist = input_data.read_data_sets(file, one_hot=True)

def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))



def upsample(input, size):
    return tf.image.resize_nearest_neighbor(input, size=(int(size), int(size)))

def conv2d(input, in_features, out_features, kernel_size, name,st=1,with_bias=False):
    W = weight_var([ kernel_size, kernel_size, in_features, out_features ],name=name)
    conv = tf.nn.conv2d(input, W, [ 1, st, st, 1 ], padding='SAME')
    if with_bias:
        return tf.add(conv , bias_var([ out_features ],name=name+'b'))
    return conv

def max_pool(input, s, st=-1):
    if(st==-1):
        st=s;
    return tf.nn.max_pool(input, [ 1, s, s, 1 ], [1, st, st, 1 ], 'SAME')

def generator(input):
    
    with tf.variable_scope("generator",reuse=tf.AUTO_REUSE):
        
#        current=conv2d(input,1,4,3)
#        current=tf.nn.relu(current)
#        current=conv2d(current,4,4,3)
#        current=tf.nn.relu(current)
#        current=upsample(current,14)
#        current=conv2d(current,4,4,3)
#        current=tf.nn.relu(current)
#        current=conv2d(current,4,4,3)
#        current=tf.nn.relu(current)
#        current=upsample(current,28)
#        current=conv2d(current,4,4,3)
#        current=tf.nn.relu(current)
#        current=conv2d(current,4,4,3)
#        current=tf.nn.relu(current)
        G_W1 = weight_var([100, 128], 'G_W1')
        G_b1 = bias_var([128], 'G_B1')

#        G_W2 = weight_var([128, 384], 'G_W2')
#        G_b2 = bias_var([384], 'G_B2')
        
        G_W3 = weight_var([128, 784], 'G_W3')
        G_b3 = bias_var([784], 'G_B3')
        
        G_h2 = tf.nn.relu(tf.matmul(input, G_W1) + G_b1) 
#        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2) 
        G_logit = tf.matmul(G_h2, G_W3) + G_b3 #batch*1
        G_prob = tf.nn.sigmoid(G_logit)
        return G_prob

def discriminator(input): 
    input=tf.reshape(input,shape=[-1,28,28,1])
    with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):  
        current=conv2d(input,1,2,3,name='D_C1')
        current=tf.nn.relu(current)
        current=conv2d(current,2,4,3,name='D_C2')
        current=tf.nn.relu(current)
        current=max_pool(current,2) #14*14*2
#        current=conv2d(current,2,4,3,name='D_C3')
#        current=tf.nn.relu(current)
#        current=conv2d(current,4,4,3,name='D_C4')
#        current=tf.nn.relu(current)
#        current=max_pool(current,2) #7*7*4
#        current=conv2d(current,4,8,3)
#        current=tf.nn.relu(current)
#        current=conv2d(current,8,8,3)
#        current=tf.nn.relu(current)
        current = tf.reshape(current, [ -1, 784 ])
        
        
        D_W1 = weight_var([784, 128], 'D_W1')
        D_b1 = bias_var([128], 'D_b1')

        D_W2 = weight_var([128, 1], 'D_W2')
        D_b2 = bias_var([1], 'D_b2')

        
        D_h1 = tf.nn.relu(tf.matmul(current, D_W1) + D_b1) #batch*64
        D_logit = tf.matmul(D_h1, D_W2) + D_b2 #batch*1
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit


def randominit(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def main():

    X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
    Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
    lr = tf.placeholder("float", shape=[], name='learning_rate')
    
    smooth=0.01
    batch_size = 256
    Z_dim = 100
    learning_rate_init=0.002
    iter_time=10000
    show_init=randominit(16, Z_dim)
    
    
    G_sample = generator(Z)
    D_real, D_logit_real = discriminator(X)
    D_fake, D_logit_fake = discriminator(G_sample)
    
    
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real))*(1-smooth))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake))*(1-smooth))
    
    train_vars = tf.trainable_variables()
    
    d_vars= [ var for var in train_vars if var.name.startswith("discriminator")]
    g_vars= [ var for var in train_vars if var.name.startswith("generator")]
    
    
    D_optimizer = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=d_vars)
    G_optimizer = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=g_vars)
    
    
    sess = tf.InteractiveSession()
    
    if not os.path.exists('out/'):
        os.makedirs('out/')
    
    sess.run(tf.global_variables_initializer())
    
    i=0
    for it in range(iter_time):
        learning_rate=learning_rate_init
        if it>=iter_time*0.2: learning_rate=learning_rate_init/2.0
        elif it>=iter_time*0.4: learning_rate=learning_rate_init/4.0
        elif it>=iter_time*0.6: learning_rate=learning_rate_init/8.0
        elif it>=iter_time*0.8: learning_rate=learning_rate_init/16.0
        
        ####中间 画图
        if it % 100 == 0:
            samples = sess.run(G_sample, feed_dict={
                               Z: show_init})  
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
        ####
        
        
        X_mb, _ = mnist.train.next_batch(batch_size)
    
        _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={lr:learning_rate,
                                  X: X_mb, Z: randominit(batch_size, Z_dim)})
    #    writer = tf.summary.FileWriter("D://TensorBoard//test",sess.graph)
        _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={lr:learning_rate,
                                  Z: randominit(batch_size, Z_dim)})
    #    writer = tf.summary.FileWriter("D://TensorBoard//test",sess.graph)
    
        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()


if __name__ == '__main__':
    main()