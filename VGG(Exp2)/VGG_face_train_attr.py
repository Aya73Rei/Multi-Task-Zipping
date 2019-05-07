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

require_improvement=2

averageImg_BGR = tf.constant(np.expand_dims(np.expand_dims(np.array([93.5940,104.7624,129.1863],dtype=np.float32),axis=0),axis=0))



#%% functions
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.cast(tf.image.decode_jpeg(image_string),dtype=tf.float32)
  image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 224,224)
  image_bgr = tf.reverse(image_resized,axis=[-1])
  image_nml = image_bgr-averageImg_BGR
  return image_nml, label

#%% load data
file_paths_face=np.array([os.getcwd()+ '/img_align_celeba/' + x for x in sorted(os.listdir('./img_align_celeba/'))])

labels_face = np.array(pd.read_csv('list_attr_celeba.txt',delim_whitespace=True,header=None).values[:,1:],dtype=np.float32)

partition_face=np.array(pd.read_csv('list_eval_partition.txt',sep=' ',header=None).values[:,1],dtype=int)

file_paths_face_train=file_paths_face[partition_face==0]
labels_face_train=labels_face[partition_face==0]

nb_exp_train=len(labels_face_train)
nb_batches=nb_exp_train//BATCH_SIZE +1


batch_rand_index=np.random.choice(len(labels_face_train),size=len(labels_face_train), replace=False)

file_paths_face_train=file_paths_face_train[batch_rand_index]
labels_face_train=labels_face_train[batch_rand_index]

file_paths_face_train=tf.constant(file_paths_face_train)
labels_face_train=tf.constant(labels_face_train)

file_paths_face=file_paths_face[np.logical_or(partition_face==1,partition_face==2)]
labels_face_val=labels_face[np.logical_or(partition_face==1,partition_face==2)]

np_exp_val=len(labels_face_val)

file_paths_face_val=tf.constant(file_paths_face)
labels_face_val=tf.constant(labels_face_val)


#%% construct input pipeline

dataset_face_train = tf.data.Dataset.from_tensor_slices((file_paths_face_train,labels_face_train))

dataset_face_train = dataset_face_train.shuffle(buffer_size=10000)
dataset_face_train = dataset_face_train.map(map_func=_parse_function, num_parallel_calls=CPU_CORES)
dataset_face_train = dataset_face_train.batch(BATCH_SIZE)
dataset_face_train = dataset_face_train.prefetch(buffer_size=1)


dataset_face_val=tf.data.Dataset.from_tensor_slices((file_paths_face_val,labels_face_val))
dataset_face_val = dataset_face_val.map(map_func=_parse_function, num_parallel_calls=CPU_CORES)
dataset_face_val = dataset_face_val.batch(BATCH_SIZE)
dataset_face_val = dataset_face_val.prefetch(buffer_size=1)

iterator = tf.data.Iterator.from_structure(dataset_face_train.output_types,
                                           dataset_face_train.output_shapes)
x, y_true=iterator.get_next()

#training_init_op = iterator.make_initializer(training_dataset)
#validation_init_op = iterator.make_initializer(validation_dataset)

#with tf.Session() as sess:
#    sess.run(iterator.initializer)
#    pic=sess.run(a)


#%% Create Weights
W_VGG_Face={
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
        'fc8':tf.get_variable('W_VGG_Face_fc8', shape=[4096,40]),
        }

b_VGG_Face={
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
        'fc8':tf.get_variable('b_VGG_Face_fc8', shape=[40]),
        }

#Weights_VGG_Face, bias_VGG_Face = pickle.load(open('Weights_VGG_Face','rb'))
Weights_VGG_Face, bias_VGG_Face = pickle.load(open('Weights_VGG_imdb','rb'))
#%% Define Model Graph
#x = tf.placeholder(tf.float32, [None, 218,178,3])
#y_true=tf.placeholder(tf.float32, [None,40])
keep_ratio = tf.placeholder(tf.float32)

conv1_1 = tf.nn.relu(tf.nn.conv2d(x, W_VGG_Face['conv1_1'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv1_1'])
conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1, W_VGG_Face['conv1_2'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv1_2'])

pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

conv2_1 = tf.nn.relu(tf.nn.conv2d(pool1, W_VGG_Face['conv2_1'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv2_1'])
conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1, W_VGG_Face['conv2_2'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv2_2'])

pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

conv3_1 = tf.nn.relu(tf.nn.conv2d(pool2, W_VGG_Face['conv3_1'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv3_1'])
conv3_2 = tf.nn.relu(tf.nn.conv2d(conv3_1, W_VGG_Face['conv3_2'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv3_2'])
conv3_3 = tf.nn.relu(tf.nn.conv2d(conv3_2, W_VGG_Face['conv3_3'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv3_3'])

pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

conv4_1 = tf.nn.relu(tf.nn.conv2d(pool3, W_VGG_Face['conv4_1'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv4_1'])
conv4_2 = tf.nn.relu(tf.nn.conv2d(conv4_1, W_VGG_Face['conv4_2'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv4_2'])
conv4_3 = tf.nn.relu(tf.nn.conv2d(conv4_2, W_VGG_Face['conv4_3'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv4_3'])

pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

conv5_1 = tf.nn.relu(tf.nn.conv2d(pool4, W_VGG_Face['conv5_1'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv5_1'])
conv5_2 = tf.nn.relu(tf.nn.conv2d(conv5_1, W_VGG_Face['conv5_2'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv5_2'])
conv5_3 = tf.nn.relu(tf.nn.conv2d(conv5_2, W_VGG_Face['conv5_3'] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face['conv5_3'])

pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

flatten5 = tf.contrib.layers.flatten(pool5)

fc6 = tf.nn.relu(tf.add(tf.matmul(flatten5, W_VGG_Face['fc6']), b_VGG_Face['fc6']))
dropout6 = tf.nn.dropout(fc6, keep_prob=keep_ratio)

fc7 = tf.nn.relu(tf.add(tf.matmul(dropout6, W_VGG_Face['fc7']), b_VGG_Face['fc7']))
dropout7 = tf.nn.dropout(fc7, keep_prob=keep_ratio)


y = tf.nn.tanh(tf.add(tf.matmul(dropout7, W_VGG_Face['fc8']), b_VGG_Face['fc8']))


#%% define loss functions
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y)

corr_pred = tf.equal(y_true, tf.sign(y))
accuracy_sum = tf.reduce_sum(tf.cast(corr_pred, tf.float32))/40

opt = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True)
opt_op=opt.minimize(loss)

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
                      'conv5_3',]

for layer_name in conv_layer_name_list:
        sess.run(tf.assign(W_VGG_Face[layer_name],Weights_VGG_Face[layer_name]))
        sess.run(tf.assign(b_VGG_Face[layer_name],bias_VGG_Face[layer_name]))

#%%
#writer = tf.summary.FileWriter('log',sess.graph)

#%% training
best_validation_accuracy=0
epoch_count=0
patience=0
#saver = tf.train.Saver()
##save_dir = 'checkpoints_single_left_unpruned/'
#save_dir = 'tf_saved_VGG_face/'
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)
#save_path = os.path.join(save_dir, 'best_validation')

print('start training')
for i in range(3):
    epoch_count += 1
    sess.run(iterator.make_initializer(dataset_face_train))
    batch_count = 0
    curr_time=time.time()
    while True:
        try:
            sess.run(opt_op, feed_dict={keep_ratio:0.5})
            batch_count += 1
            if batch_count % 10 ==0:
                print('epoch={:d},batch={:d}/{:d},used_time:{:.2f}s'.format(epoch_count,batch_count,nb_batches,time.time()-curr_time))
                curr_time=time.time()
        except tf.errors.OutOfRangeError:
            break

    sess.run(iterator.make_initializer(dataset_face_val))
    acc_sum=0
    while True:
        try:
            batch_count += 1
            acc_sum += sess.run(accuracy_sum, feed_dict={keep_ratio:1})
        except tf.errors.OutOfRangeError:
            break
    acc_val=acc_sum/np_exp_val
    print('\nEpoch:{:d}, val_acc={:%}'.format(epoch_count, acc_val))
#%%
#saver.restore(sess, save_path)
W_dict=sess.run(W_VGG_Face)
b_dict=sess.run(b_VGG_Face)

#config_tuple=open(save_dir+'unpruned_weights','wb')
#pickle.dump((W_dict, b_dict), config_tuple)
#config_tuple.close()
