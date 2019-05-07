#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 02:15:55 2018

@author: hex
"""

import tensorflow as tf
import numpy as np
import os
import pandas as pd

#%% hyper params

#%%
def _parse_function_VGG_Face(filename, label):
    averageImg_BGR_VGG_Face = tf.constant(np.expand_dims(np.expand_dims(np.array([93.5940,104.7624,129.1863],dtype=np.float32),axis=0),axis=0))
    image_string = tf.read_file(filename)
    image_decoded = tf.cast(tf.image.decode_jpeg(image_string),dtype=tf.float32)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 224,224)
    image_bgr = tf.reverse(image_resized,axis=[-1])
    image_nml = image_bgr-averageImg_BGR_VGG_Face
    return image_nml, label

def load_celebA(BATCH_SIZE, CPU_CORES):
    file_paths_face=np.array([os.getcwd()+ '/img_align_celeba/' + x for x in sorted(os.listdir('./img_align_celeba/'))])

    labels_face = np.array(pd.read_csv('list_attr_celeba.txt',delim_whitespace=True,header=None).values[:,1:],dtype=np.float32)

    labels_face[labels_face==-1]=0

    partition_face=np.array(pd.read_csv('list_eval_partition.txt',sep=' ',header=None).values[:,1],dtype=int)

    file_paths_face_train=file_paths_face[partition_face==0]
    labels_face_train=labels_face[partition_face==0]

    nb_exp_train=len(labels_face_train)


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

    dataset_face_train = tf.data.Dataset.from_tensor_slices((file_paths_face_train,labels_face_train))
    dataset_face_train = dataset_face_train.shuffle(buffer_size=100000)
    dataset_face_train = dataset_face_train.map(map_func=_parse_function_VGG_Face, num_parallel_calls=CPU_CORES)
    dataset_face_train = dataset_face_train.batch(BATCH_SIZE)
    dataset_face_train = dataset_face_train.prefetch(buffer_size=1)


    dataset_face_val=tf.data.Dataset.from_tensor_slices((file_paths_face_val,labels_face_val))
#    dataset_face_val = dataset_face_val.shuffle(buffer_size=100000)
    dataset_face_val = dataset_face_val.map(map_func=_parse_function_VGG_Face, num_parallel_calls=CPU_CORES)
    dataset_face_val = dataset_face_val.batch(BATCH_SIZE)
    dataset_face_val = dataset_face_val.prefetch(buffer_size=1)

    return dataset_face_train, dataset_face_val, nb_exp_train, np_exp_val

#%%
def _parse_function_imageNet(filename, label):
    averageImg_BGR_imageNet = tf.constant(np.expand_dims(np.expand_dims(np.array([103.939, 116.779, 123.68],dtype=np.float32),axis=0),axis=0))
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
    image_nml = image_bgr-averageImg_BGR_imageNet
    label=tf.one_hot(indices=label,depth=1000)
    return image_nml, label

def _parse_function_imageNet_train(filename, label):
    averageImg_BGR_imageNet = tf.constant(np.expand_dims(np.expand_dims(np.array([103.939, 116.779, 123.68],dtype=np.float32),axis=0),axis=0))
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
#    image_resized = tf.image.resize_images(image_decoded, [224, 224])
#    tf.Print(image_resized,[tf.shape(image_resized)],message='shape=')
    image_bgr = tf.reverse(image_resized,axis=[-1])
    image_nml = image_bgr-averageImg_BGR_imageNet
    image_cropped = tf.image.random_flip_left_right(tf.random_crop(image_nml, [224,224,3]))
    label=tf.one_hot(indices=label,depth=1000)
    return image_cropped, label

def load_imageNet(BATCH_SIZE, CPU_CORES):
#% load data
    file_paths_imageNet=np.array([os.getcwd()+ '/ILSVRC2012_img_val/' + x for x in sorted(os.listdir('./ILSVRC2012_img_val/'))])

    labels_imageNet=np.array(pd.read_csv('ILSVRC_labels.txt', delim_whitespace=True,header=None).values[:,1],dtype=np.int32)

    train_index_imageNet=np.ones(labels_imageNet.shape,dtype=np.bool)
    val_index_imageNet=np.zeros(labels_imageNet.shape,dtype=np.bool)

    for i in range(1000):
        class_index=np.argwhere(labels_imageNet==i)
#        rand_index=np.random.choice(50,10, replace=False)
        rand_index=[25,  0, 24,  8, 37, 19,  3, 14, 15, 38]
        train_index_imageNet[class_index[rand_index]]=False
        val_index_imageNet[class_index[rand_index]]=True


    file_paths_imageNet_train=file_paths_imageNet[train_index_imageNet]
    labels_imageNet_train=labels_imageNet[train_index_imageNet]

    nb_exp_train=len(labels_imageNet_train)


    file_paths_imageNet_val=file_paths_imageNet[val_index_imageNet]
    labels_imageNet_val=labels_imageNet[val_index_imageNet]

    np_exp_val=len(labels_imageNet_val)

    file_paths_imageNet_train=tf.constant(file_paths_imageNet_train)
    labels_imageNet_train=tf.constant(labels_imageNet_train)

    file_paths_imageNet_val=tf.constant(file_paths_imageNet_val)
    labels_imageNet_val=tf.constant(labels_imageNet_val)


#%% construct input pipeline

    dataset_imageNet_train = tf.data.Dataset.from_tensor_slices((file_paths_imageNet_train,labels_imageNet_train))
    dataset_imageNet_train = dataset_imageNet_train.shuffle(buffer_size=100000)
    dataset_imageNet_train = dataset_imageNet_train.repeat(count=4)
    dataset_imageNet_train = dataset_imageNet_train.map(map_func=_parse_function_imageNet_train, num_parallel_calls=CPU_CORES)
    dataset_imageNet_train = dataset_imageNet_train.batch(BATCH_SIZE)
    dataset_imageNet_train = dataset_imageNet_train.prefetch(buffer_size=1)


    dataset_imageNet_val=tf.data.Dataset.from_tensor_slices((file_paths_imageNet_val,labels_imageNet_val))
#    dataset_imageNet_val = dataset_imageNet_val.shuffle(buffer_size=100000)
    dataset_imageNet_val = dataset_imageNet_val.map(map_func=_parse_function_imageNet, num_parallel_calls=CPU_CORES)
    dataset_imageNet_val = dataset_imageNet_val.batch(BATCH_SIZE)
    dataset_imageNet_val = dataset_imageNet_val.prefetch(buffer_size=1)

    return dataset_imageNet_train, dataset_imageNet_val, nb_exp_train, np_exp_val