#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 02:22:20 2018

@author: hex
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import numpy as np
import time
from datetime import timedelta, datetime
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import matplotlib.pyplot as plt
import copy
import pandas as pd
tf.reset_default_graph()

#%% hyper params

BATCH_SIZE=128
CPU_CORES=8

#require_improvement=3

averageImg_BGR = tf.constant(np.expand_dims(np.expand_dims(np.array([103.939, 116.779, 123.68],dtype=np.float32),axis=0),axis=0))



#%% functions
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.cast(tf.image.decode_jpeg(image_string),dtype=tf.float32)
  h=tf.cast(tf.shape(image_decoded)[0],dtype=tf.float32)
  w=tf.cast(tf.shape(image_decoded)[1],dtype=tf.float32)
  def f1(): return tf.cast(256/w*h,dtype=tf.int32)
  def f2(): return tf.cast(256/h*w,dtype=tf.int32)
  def f3(): return tf.constant(256)
  h_r=tf.case({h<=w:f3,h>w:f1},exclusive=True)
  w_r=tf.case({w<=h:f3,w>h:f2},exclusive=True)
  image_resized = tf.image.resize_images(image_decoded, [h_r, w_r])
  image_cropped = tf.image.resize_image_with_crop_or_pad(image_resized, 224,224)
  image_bgr = tf.reverse(image_cropped,axis=[-1])
  image_nml = image_bgr-averageImg_BGR
  return image_nml, label

#%% load data
file_paths_imageNet=np.array([os.getcwd()+ '/ILSVRC2012_img_val/' + x for x in sorted(os.listdir('./ILSVRC2012_img_val/'))])

labels_imageNet=np.array(pd.read_csv('ILSVRC_labels.txt', delim_whitespace=True,header=None).values[:,1],dtype=np.int32)

train_index_imageNet=np.ones(labels_imageNet.shape,dtype=np.bool)
val_index_imageNet=np.zeros(labels_imageNet.shape,dtype=np.bool)

for i in range(1000):
    class_index=np.argwhere(labels_imageNet==i)
    rand_index=np.random.choice(50,10, replace=False)
    train_index_imageNet[class_index[rand_index]]=False
    val_index_imageNet[class_index[rand_index]]=True


file_paths_imageNet_train=file_paths_imageNet[train_index_imageNet]
labels_imageNet_train=labels_imageNet[train_index_imageNet]

nb_exp_train=len(labels_imageNet_train)
nb_batches=nb_exp_train//BATCH_SIZE +1


file_paths_imageNet_val=file_paths_imageNet[val_index_imageNet]
labels_imageNet_val=labels_imageNet[val_index_imageNet]

np_exp_val=len(labels_imageNet_val)

file_paths_imageNet_train=tf.constant(file_paths_imageNet_train)
labels_imageNet_train=tf.constant(labels_imageNet_train)

file_paths_imageNet_val=tf.constant(file_paths_imageNet_val)
labels_imageNet_val=tf.constant(labels_imageNet_val)


#%% construct input pipeline

dataset_imageNet_train = tf.data.Dataset.from_tensor_slices((file_paths_imageNet_train,labels_imageNet_train))

dataset_imageNet_train = dataset_imageNet_train.shuffle(buffer_size=10000)
dataset_imageNet_train = dataset_imageNet_train.map(map_func=_parse_function, num_parallel_calls=CPU_CORES)
dataset_imageNet_train = dataset_imageNet_train.batch(BATCH_SIZE)
dataset_imageNet_train = dataset_imageNet_train.prefetch(buffer_size=1)


dataset_imageNet_val=tf.data.Dataset.from_tensor_slices((file_paths_imageNet_val,labels_imageNet_val))
dataset_imageNet_val = dataset_imageNet_val.map(map_func=_parse_function, num_parallel_calls=CPU_CORES)
dataset_imageNet_val = dataset_imageNet_val.batch(BATCH_SIZE)
dataset_imageNet_val = dataset_imageNet_val.prefetch(buffer_size=1)

iterator = tf.data.Iterator.from_structure(dataset_imageNet_train.output_types,
                                           dataset_imageNet_train.output_shapes)
x, y_true=iterator.get_next()

image_string = tf.read_file(tf.constant(file_paths_imageNet[13]))
image_decoded = tf.cast(tf.image.decode_jpeg(image_string),dtype=tf.float32)
#image_resized = tf.image.resize_images(image_decoded, [224, 224])
#image_bgr = tf.reverse(image_resized,axis=[-1])
#image_nml = image_bgr-averageImg_BGR
#x=tf.expand_dims(image_nml,axis=0)
#x=tf.constant(image)
#training_init_op = iterator.make_initializer(training_dataset)
#validation_init_op = iterator.make_initializer(validation_dataset)

#with tf.Session() as sess:
#    sess.run(iterator.initializer)
#    pic=sess.run(a)


#%% Create Weights
W_imageNet={
        'conv1_1':tf.get_variable('W_VGG_Face_conv1_1', shape=[3,3,3,64],trainable=False),
        'conv1_2':tf.get_variable('W_VGG_Face_conv1_2', shape=[3,3,64,64],trainable=False),
        'conv2_1':tf.get_variable('W_VGG_Face_conv2_1', shape=[3,3,64,128],trainable=False),
        'conv2_2':tf.get_variable('W_VGG_Face_conv2_2', shape=[3,3,128,128],trainable=False),
        'conv3_1':tf.get_variable('W_VGG_Face_conv3_1', shape=[3,3,128,256],trainable=False),
        'conv3_2':tf.get_variable('W_VGG_Face_conv3_2', shape=[3,3,256,256],trainable=False),
        'conv3_3':tf.get_variable('W_VGG_Face_conv3_3', shape=[3,3,256,256],trainable=False),
        'conv4_1':tf.get_variable('W_VGG_Face_conv4_1', shape=[3,3,256,512],trainable=False),
        'conv4_2':tf.get_variable('W_VGG_Face_conv4_2', shape=[3,3,512,512],trainable=False),
        'conv4_3':tf.get_variable('W_VGG_Face_conv4_3', shape=[3,3,512,512],trainable=False),
        'conv5_1':tf.get_variable('W_VGG_Face_conv5_1', shape=[3,3,512,512],trainable=False),
        'conv5_2':tf.get_variable('W_VGG_Face_conv5_2', shape=[3,3,512,512],trainable=False),
        'conv5_3':tf.get_variable('W_VGG_Face_conv5_3', shape=[3,3,512,512],trainable=False),
        'fc6':tf.get_variable('W_VGG_Face_fc6', shape=[25088,4096]),
        'fc7':tf.get_variable('W_VGG_Face_fc7', shape=[4096,4096]),
        'fc8':tf.get_variable('W_VGG_Face_fc8', shape=[4096,1000]),
        }

b_imageNet={
        'conv1_1':tf.get_variable('b_VGG_Face_conv1_1', shape=[64],trainable=False),
        'conv1_2':tf.get_variable('b_VGG_Face_conv1_2', shape=[64],trainable=False),
        'conv2_1':tf.get_variable('b_VGG_Face_conv2_1', shape=[128],trainable=False),
        'conv2_2':tf.get_variable('b_VGG_Face_conv2_2', shape=[128],trainable=False),
        'conv3_1':tf.get_variable('b_VGG_Face_conv3_1', shape=[256],trainable=False),
        'conv3_2':tf.get_variable('b_VGG_Face_conv3_2', shape=[256],trainable=False),
        'conv3_3':tf.get_variable('b_VGG_Face_conv3_3', shape=[256],trainable=False),
        'conv4_1':tf.get_variable('b_VGG_Face_conv4_1', shape=[512],trainable=False),
        'conv4_2':tf.get_variable('b_VGG_Face_conv4_2', shape=[512],trainable=False),
        'conv4_3':tf.get_variable('b_VGG_Face_conv4_3', shape=[512],trainable=False),
        'conv5_1':tf.get_variable('b_VGG_Face_conv5_1', shape=[512],trainable=False),
        'conv5_2':tf.get_variable('b_VGG_Face_conv5_2', shape=[512],trainable=False),
        'conv5_3':tf.get_variable('b_VGG_Face_conv5_3', shape=[512],trainable=False),
        'fc6':tf.get_variable('b_VGG_Face_fc6', shape=[4096]),
        'fc7':tf.get_variable('b_VGG_Face_fc7', shape=[4096]),
        'fc8':tf.get_variable('b_VGG_Face_fc8', shape=[1000]),
        }

Weights_imageNet, bias_imageNet = pickle.load(open('Weights_imageNet','rb'))
#%% Define Model Graph
#x = tf.placeholder(tf.float32, [None, 218,178,3])
#y_true=tf.placeholder(tf.float32, [None,40])
keep_ratio = tf.placeholder(tf.float32)

conv1_1 = tf.nn.relu(tf.nn.conv2d(x, W_imageNet['conv1_1'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv1_1'])
conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1, W_imageNet['conv1_2'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv1_2'])

pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

conv2_1 = tf.nn.relu(tf.nn.conv2d(pool1, W_imageNet['conv2_1'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv2_1'])
conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1, W_imageNet['conv2_2'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv2_2'])

pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

conv3_1 = tf.nn.relu(tf.nn.conv2d(pool2, W_imageNet['conv3_1'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv3_1'])
conv3_2 = tf.nn.relu(tf.nn.conv2d(conv3_1, W_imageNet['conv3_2'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv3_2'])
conv3_3 = tf.nn.relu(tf.nn.conv2d(conv3_2, W_imageNet['conv3_3'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv3_3'])


pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

conv4_1 = tf.nn.relu(tf.nn.conv2d(pool3, W_imageNet['conv4_1'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv4_1'])
conv4_2 = tf.nn.relu(tf.nn.conv2d(conv4_1, W_imageNet['conv4_2'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv4_2'])
conv4_3 = tf.nn.relu(tf.nn.conv2d(conv4_2, W_imageNet['conv4_3'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv4_3'])

pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

conv5_1 = tf.nn.relu(tf.nn.conv2d(pool4, W_imageNet['conv5_1'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv5_1'])
conv5_2 = tf.nn.relu(tf.nn.conv2d(conv5_1, W_imageNet['conv5_2'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv5_2'])
conv5_3 = tf.nn.relu(tf.nn.conv2d(conv5_2, W_imageNet['conv5_3'] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet['conv5_3'])

pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')
#flatten5 = tf.contrib.layers.flatten(tf.transpose(pool5, [0,3,1,2]))
flatten5 = tf.contrib.layers.flatten(pool5)

fc6 = tf.nn.relu(tf.add(tf.matmul(flatten5, W_imageNet['fc6']), b_imageNet['fc6']))
dropout6 = tf.nn.dropout(fc6, keep_prob=keep_ratio)

fc7 = tf.nn.relu(tf.add(tf.matmul(dropout6, W_imageNet['fc7']), b_imageNet['fc7']))
dropout7 = tf.nn.dropout(fc7, keep_prob=keep_ratio)

y = tf.add(tf.matmul(dropout7, W_imageNet['fc8']), b_imageNet['fc8'])


#%% define loss functions
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y)
loss = tf.reduce_mean(cross_entropy)

top1_sum = tf.reduce_sum(tf.cast(tf.nn.in_top_k(predictions=y, targets=y_true, k=1),dtype=tf.int32))
top5_sum = tf.reduce_sum(tf.cast(tf.nn.in_top_k(predictions=y, targets=y_true, k=5),dtype=tf.int32))
#opt = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True)
#opt_op=opt.minimize(loss)

#%%
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

sess = tf.Session()

#%%
sess.run(tf.global_variables_initializer())

conv_layer_name_list=['conv1_1',
                      'conv1_2',
                      'conv2_1',
                      'conv2_2',
                      'conv3_1',
                      'conv3_2',
                      'conv3_3',
                      'conv4_1',
                      'conv4_2',
                      'conv4_3',
                      'conv5_1',
                      'conv5_2',
                      'conv5_3',
                      'fc6',
                      'fc7',
                      'fc8']

for layer_name in conv_layer_name_list:
        sess.run(tf.assign(W_imageNet[layer_name],Weights_imageNet[layer_name]))
        sess.run(tf.assign(b_imageNet[layer_name],bias_imageNet[layer_name]))

#%%
#writer = tf.summary.FileWriter('log',sess.graph)

#%% training
#best_validation_accuracy=0
#epoch_count=0
#patience=0
#saver = tf.train.Saver()
#save_dir = 'checkpoints_single_left_unpruned/'
#save_dir = 'tf_saved_VGG_face/'
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)
#save_path = os.path.join(save_dir, 'best_validation')

sess.run(iterator.make_initializer(dataset_imageNet_val))
acc_1_sum=0
acc_5_sum=0
batch_count=0
curr_time=time.time()
while True:
    try:
        batch_count += 1
        tmp1,tmp2= sess.run((top1_sum,top5_sum), feed_dict={keep_ratio:1})
        acc_1_sum+=tmp1
        acc_5_sum+=tmp2
        if batch_count % 10 ==0:
            print('\rbatch={:d}/{:d},used_time:{:.2f}s'.format(batch_count,nb_batches,time.time()-curr_time),end=' ')
            curr_time=time.time()
    except tf.errors.OutOfRangeError:
        break
acc_val_top1=acc_1_sum/np_exp_val
acc_val_top5=acc_5_sum/np_exp_val
#%%

#a=sess.run(tf.argmax(y, axis=1, output_type=tf.int32), feed_dict={keep_ratio:1})
#b=sess.run(pool5, feed_dict={keep_ratio:1})
#
#W_dict=sess.run(W_imageNet)
#b_dict=sess.run(b_imageNet)

