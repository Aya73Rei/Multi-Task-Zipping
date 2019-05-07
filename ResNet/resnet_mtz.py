#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 02:01:35 2018

@author: hex
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wangxu on 2018/9/27
import tensorflow as tf
import time
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
#import logging
#import pandas as pd

tf.reset_default_graph()

class ResNet_MTZ:
    def __init__(self, dataset_path='/home/hex/data/', model_path=[], weights_folder='model_weights/'):
        self.reset_graph()
        self.batch_size = 64
        self.cpu_cores = 8
        self.model_path = dict()
        self.dataset_path = dataset_path
        self.n_group = 3
        self.n_blocks_per_group = 4
        self.width = 32
        self.task_name_list = []
        self.weights_folder = weights_folder

        self.n_classes = dict()
        self.n_sample_train = dict()
        self.total_batches_train = dict()
        self.n_samples_val = dict()
        self.X = dict()
        self.Y = dict()
        self.train_init = dict()
        self.test_init = dict()
        self.hessian_init = dict()
        self.loss = dict()
        self.accuracy = dict()
        self.logits = dict()
        self.opt = dict()
        self.opt_op = dict()
        self.merged_task_list=[]
        self.layer_input = dict()
        self.lr = 0.001

        if not model_path:
            self.weight_dict = dict()
            self.is_merged_dict = dict()
            print('the model is current empty')
        else:
            self.weight_dict, self.is_merged_dict, self.merged_task_list= pickle.load(open(model_path, 'rb'))
            print("loading weight matrix")
            for tmp in self.merged_task_list:
                self.construct_model(tmp)

#        self.layer_output = dict()
#        self.gstep = tf.Variable(0, dtype=tf.int32,
#                                 trainable=False, name='global_step')

    def add_new_task(self, task_name):
        print('Loading task {:s}'.format(task_name))
        self.weight_dict[task_name] = pickle.load(open(self.weights_folder+task_name, 'rb'))
        self.task_name_list.append(task_name)
        self.is_merged_dict[task_name] = self.construct_is_merged_dict()

        self.construct_model(task_name)

    def construct_is_merged_dict(self):
        is_merged_dict=dict()
        is_merged_dict['pre_conv'] = False
        for i in range(self.n_group):
                for j in range(self.n_blocks_per_group):
                    block_name = 'conv{:d}_{:d}'.format(i+1,j+1)
                    is_merged_dict[block_name+'/conv_1'] = False
                    is_merged_dict[block_name+'/conv_2'] = False

        return is_merged_dict

    def merge_initial_tasks(self, task_A, task_B):
        self.weight_dict['shared'] = dict()
        layer_list = list(self.weight_dict[task_A].keys())[:-2]
        self.merged_task_list.append(task_A)
        self.merged_task_list.append(task_B)
        layer_count = 0
        for layer_name in layer_list:
            self.weight_dict['shared'][layer_name] = dict()
            print('Merging layer {:s}'.format(layer_name))
            self.merge_weights_at_layer(layer_name, task_A, task_B, merge_degree=0.9)
            self.reset_graph()
            self.is_merged_dict[task_A][layer_name] = True
            self.is_merged_dict[task_B][layer_name] = True
            self.construct_model(task_A)
            self.construct_model(task_B)
            self.build()
#            self.test_merged()
            self.joint_finetune(steps=10+layer_count*7)
            layer_count += 1
#            self.test_merged()
            self.weight_dict=self.fetch_weight()
        self.test_merged()

    def merge_new_task(self, task_name):
        print('merging new task {:s}'.format(task_name))
        layer_list = list(self.weight_dict['shared'].keys())
        self.merged_task_list.append(task_name)
        layer_count = 0
        for layer_name in layer_list:
            print('Merging layer {:s}'.format(layer_name))
            self.merge_weights_at_layer_add(layer_name, task_name)
            self.reset_graph()
            self.is_merged_dict[task_name][layer_name] = True
            for tmp in self.merged_task_list:
                self.construct_model(tmp)
            self.build()
#            self.test_merged()
            self.joint_finetune(steps=10+layer_count*7)
            layer_count += 1
#            self.test_merged()
            self.weight_dict=self.fetch_weight()
        self.test_merged()

    def reset_graph(self):
        tf.reset_default_graph()
        self.training = tf.placeholder(tf.bool, name='training')
        self.regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.0001)
        self.regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.0005)

    def merge_weights_at_layer_add(self, layer_name, task_name):
        weights_new_task = self.weight_dict[task_name][layer_name]['weights']

        merge_list_all, merged_vector = self.generate_merge_list_conv_add(task_name, layer_name)
        tomerge_nb = self.weight_dict['shared'][layer_name]['weights'].shape[-1]

        merge_list = merge_list_all[:tomerge_nb]

        merge_list = merge_list[np.argsort(merge_list[:,0])]

        mask = np.ones(weights_new_task.shape[3],dtype=bool)

        mask[merge_list[:,1]] = False

        self.weight_dict['shared'][layer_name]['weights'] = np.float32(merged_vector[:,:,:,merge_list[:,0],merge_list[:,1]])

        self.weight_dict[task_name][layer_name]['weights'] = np.float32(self.weight_dict[task_name][layer_name]['weights'][:,:,:,mask])

        permindex = np.concatenate((merge_list[:,1], np.arange(weights_new_task.shape[3])[mask]))

        self.weight_dict[task_name][layer_name]['permutation'] = np.int32(np.argsort(permindex))

    def merge_weights_at_layer(self, layer_name, task_A, task_B, merge_degree=0.9):
        weights_A = self.weight_dict[task_A][layer_name]['weights']
        weights_B = self.weight_dict[task_B][layer_name]['weights']

        merge_list_all, merged_vector = self.generate_merge_list_conv(task_A, task_B, layer_name)
        n_filter_A = self.weight_dict[task_A][layer_name]['weights'].shape[-1]


        tomerge_nb = np.int(np.floor(n_filter_A*merge_degree))
        merge_list = merge_list_all[:tomerge_nb]

        mask_A = np.ones(weights_A.shape[3],dtype=bool)
        mask_B = np.ones(weights_B.shape[3],dtype=bool)
        mask_A[merge_list[:,0]] = False
        mask_B[merge_list[:,1]] = False

        self.weight_dict['shared'][layer_name]['weights'] = np.float32(merged_vector[:,:,:,merge_list[:,0],merge_list[:,1]])

        self.weight_dict[task_A][layer_name]['weights'] = np.float32(self.weight_dict[task_A][layer_name]['weights'][:,:,:,mask_A])
        self.weight_dict[task_B][layer_name]['weights'] = np.float32(self.weight_dict[task_B][layer_name]['weights'][:,:,:,mask_B])

        permindex_A=np.concatenate((merge_list[:,0], np.arange(weights_A.shape[3])[mask_A]))
        permindex_B=np.concatenate((merge_list[:,1], np.arange(weights_B.shape[3])[mask_B]))

        self.weight_dict[task_A][layer_name]['permutation'] = np.int32(np.argsort(permindex_A))
        self.weight_dict[task_B][layer_name]['permutation'] = np.int32(np.argsort(permindex_B))

    def generate_merge_list_conv_add(self, task_name, layer_name):
        weights_A = self.weight_dict['shared'][layer_name]['weights']
        weights_B = self.weight_dict[task_name][layer_name]['weights']

        if weights_A.shape[2] != weights_B.shape[2]:
            raise ValueError('input of to be merged layers do not match')

        for i in range(len(self.merged_task_list[:-1])):
            if i == 0:
                hessian_inverse_A = np.linalg.pinv(self.calculate_hessian_conv_tf(self.merged_task_list[i], layer_name))
            else:
                hessian_inverse_A += np.linalg.pinv(self.calculate_hessian_conv_tf(self.merged_task_list[i], layer_name))

        hessian_inverse_B=np.linalg.pinv(self.calculate_hessian_conv_tf(task_name, layer_name))

        H_hat=np.linalg.pinv(hessian_inverse_A+hessian_inverse_B)
        MtxA=np.dot(hessian_inverse_A,H_hat)

        err_mtx=np.zeros((weights_A.shape[3],weights_B.shape[3],weights_A.shape[0],weights_A.shape[0]))
        merged_vector=np.zeros((weights_A.shape[0],weights_A.shape[0],weights_A.shape[2],weights_A.shape[3],weights_B.shape[3]))

        for i in range(weights_A.shape[3]):
            for j in range(weights_B.shape[3]):
                for kernel_i in range(weights_A.shape[0]):
                    for kernel_j in range(weights_B.shape[0]):
                        dist_vec=weights_A[kernel_i,kernel_j,:,i]-weights_B[kernel_i,kernel_j,:,j]
                        delta_A=np.dot(MtxA,-dist_vec)
                        merged_vector[kernel_i,kernel_j,:,i,j]=weights_A[kernel_i,kernel_j,:,i]+delta_A
                        err_mtx[i,j,kernel_i,kernel_j]=0.5*np.dot(np.dot(dist_vec,H_hat),dist_vec)

        err_mtx=np.sum(err_mtx,axis=(2,3))
#        plt.hist(np.nan_to_num(err_mtx.flatten()), bins=500)
#        plt.show()

        sorted_index = np.argsort(err_mtx,axis=None)

        merge_list_all=[]
        merge_loss=[]
        count=0
        i=0

        for i in range(weights_A.shape[3]*weights_B.shape[3]):
            l_index=sorted_index[i]//weights_B.shape[3]
            r_index=sorted_index[i]%weights_B.shape[3]
            if count==0:
               merge_list_all.append((l_index,r_index))
               merge_loss.append(err_mtx[l_index, r_index])
               count+=1
            elif np.all(np.asarray(merge_list_all)[:,0]!=l_index) and np.all(np.asarray(merge_list_all)[:,1]!=r_index):
               merge_list_all.append((l_index,r_index))
               merge_loss.append(err_mtx[l_index, r_index])
               count+=1
            #       print('i={:d}'.format(i))

#        plt.plot(merge_loss)
#        plt.show()

        return np.asarray(merge_list_all), merged_vector


    def generate_merge_list_conv(self, task_A, task_B, layer_name):
        weights_A = self.weight_dict[task_A][layer_name]['weights']
        weights_B = self.weight_dict[task_B][layer_name]['weights']

        if weights_A.shape[2] != weights_B.shape[2]:
            raise ValueError('input of to be merged layers do not match')

        hessian_inverse_A=np.linalg.pinv(self.calculate_hessian_conv_tf(task_A, layer_name))
        hessian_inverse_B=np.linalg.pinv(self.calculate_hessian_conv_tf(task_B, layer_name))

        H_hat=np.linalg.pinv(hessian_inverse_A+hessian_inverse_B)
        MtxA=np.dot(hessian_inverse_A,H_hat)

        err_mtx=np.zeros((weights_A.shape[3],weights_B.shape[3],weights_A.shape[0],weights_A.shape[0]))
        merged_vector=np.zeros((weights_A.shape[0],weights_A.shape[0],weights_A.shape[2],weights_A.shape[3],weights_B.shape[3]))

        for i in range(weights_A.shape[3]):
            for j in range(weights_B.shape[3]):
                for kernel_i in range(weights_A.shape[0]):
                    for kernel_j in range(weights_B.shape[0]):
                        dist_vec=weights_A[kernel_i,kernel_j,:,i]-weights_B[kernel_i,kernel_j,:,j]
                        delta_A=np.dot(MtxA,-dist_vec)
                        merged_vector[kernel_i,kernel_j,:,i,j]=weights_A[kernel_i,kernel_j,:,i]+delta_A
                        err_mtx[i,j,kernel_i,kernel_j]=0.5*np.dot(np.dot(dist_vec,H_hat),dist_vec)

        err_mtx=np.sum(err_mtx,axis=(2,3))
#        plt.hist(np.nan_to_num(err_mtx.flatten()), bins=500)
#        plt.show()

        sorted_index = np.argsort(err_mtx,axis=None)

        merge_list_all=[]
        merge_loss=[]
        count=0
        i=0

        for i in range(weights_A.shape[3]*weights_B.shape[3]):
            l_index=sorted_index[i]//weights_B.shape[3]
            r_index=sorted_index[i]%weights_B.shape[3]
            if count==0:
               merge_list_all.append((l_index,r_index))
               merge_loss.append(err_mtx[l_index, r_index])
               count+=1
            elif np.all(np.asarray(merge_list_all)[:,0]!=l_index) and np.all(np.asarray(merge_list_all)[:,1]!=r_index):
               merge_list_all.append((l_index,r_index))
               merge_loss.append(err_mtx[l_index, r_index])
               count+=1
            #       print('i={:d}'.format(i))

#        plt.plot(merge_loss)
#        plt.show()

        return np.asarray(merge_list_all), merged_vector

    def calculate_hessian_conv_tf(self, task_name, layer_name):
        a = tf.expand_dims(self.layer_input[task_name][layer_name], axis=-1)
        b = tf.expand_dims(self.layer_input[task_name][layer_name], axis=3)
        outprod = tf.reduce_sum(tf.reduce_mean(tf.multiply(a, b), axis=[1, 2]),axis=[0])

        batch_count = 0
        print('start calculating hessian of '+task_name)
        self.sess.run(self.hessian_init[task_name])
        try:
            while True:
                if batch_count == 0:
                    hessian_sum = self.sess.run(outprod, feed_dict={'training:0':False})
                else:
                    hessian_sum += self.sess.run(outprod, feed_dict={'training:0':False})
                batch_count += 1
#                print('{:d}'.format(batch_count))
#                if batch_count+1 % 5 ==0:
#                    print('\rbatch={:d}/{:d}'.format(batch_count+1,self.total_batches_train[task_name]),end=' ')
        except tf.errors.OutOfRangeError:
            pass

        hessian = hessian_sum/self.n_sample_train[task_name]
        return hessian

    def construct_model(self, task_name):
        self.load_dataset(task_name)

        self.layer_input[task_name] = dict()
#        with tf.variable_scope(task_name):
        self.layer_input[task_name]['pre_conv'] = self.X[task_name]
        if self.is_merged_dict[task_name]['pre_conv']:
            y = self.conv_layer_basic_merged(self.X[task_name], "pre_conv", task_name)
        else:
            y = self.conv_layer_basic(self.X[task_name], "pre_conv", task_name)

        for i in range(self.n_group):
            for j in range(self.n_blocks_per_group):
                block_name = 'conv{:d}_{:d}'.format(i+1,j+1)
                scale_down = j==0
                self.layer_input[task_name][block_name+'/conv_1'] = y
                y, self.layer_input[task_name][block_name+'/conv_2']= self.res_block(y, block_name, task_name, scale_down=scale_down)

        y = self.bn_layer(y, 'end_bn', task_name)
        y = tf.nn.relu(y)
        y = tf.reduce_mean(y,axis=[1,2])
        self.logits[task_name] = self.fc_layer(y, 'classifier', task_name)

        self.construct_loss(task_name)
        self.construct_evalutaion(task_name)
        self.construct_optimizer(task_name)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer_basic(self, bottom, layer_name, task_name, stride=1):
        weights , beta, mean, variance = self.get_conv_filter_bn(task_name, layer_name)

        conv = tf.nn.conv2d(bottom, weights , [1, stride, stride, 1], padding='SAME')
        with tf.variable_scope(task_name+'/'+layer_name):
            bn = tf.layers.batch_normalization(conv, momentum=0.1, epsilon=1e-05, training = self.training, beta_initializer=beta, scale=False, moving_mean_initializer=mean, moving_variance_initializer=variance, beta_regularizer=self.regularizer_conv)

        return bn

    def conv_layer_basic_merged(self, bottom, layer_name, task_name, stride=1):
        weights_specific, beta, mean, variance, permutation = self.get_conv_filter_bn_specific(task_name, layer_name)
        weights_shared = self.get_conv_filter_shared(layer_name)

        weights = tf.concat([weights_shared,weights_specific], axis = -1)

        conv = tf.nn.conv2d(bottom, weights , [1, stride, stride, 1], padding='SAME')
        with tf.variable_scope(task_name+'/'+layer_name):
            bn = tf.layers.batch_normalization(conv, momentum=0.1, epsilon=1e-05, training = self.training, beta_initializer=beta, scale=False, moving_mean_initializer=mean, moving_variance_initializer=variance, beta_regularizer=self.regularizer_conv)

        pmt = tf.gather(bn, permutation, axis=-1)

        return pmt

    def bn_layer(self, bottom, layer_name, task_name):
        beta, mean, variance = self.get_bn_param(task_name, layer_name)

        with tf.variable_scope(task_name+'/'+layer_name):
            bn = tf.layers.batch_normalization(bottom, momentum=0.1, epsilon=1e-05, training = self.training, beta_initializer=beta, scale=False, moving_mean_initializer=mean, moving_variance_initializer=variance, beta_regularizer=self.regularizer_conv)

        return bn

    def res_block(self, bottom, block_name, task_name, scale_down=False):
#        with tf.variable_scope(block_name):
        if scale_down:
            stride = 2
        else:
            stride = 1
        if self.is_merged_dict[task_name][block_name+'/conv_1']:
            conv_1 = self.conv_layer_basic_merged(bottom, block_name+'/conv_1', task_name, stride=stride)
        else:
            conv_1 = self.conv_layer_basic(bottom, block_name+'/conv_1', task_name, stride=stride)
        conv_1_relu = tf.nn.relu(conv_1)

        if self.is_merged_dict[task_name][block_name+'/conv_2']:
            conv_2 = self.conv_layer_basic_merged(conv_1_relu, block_name+'/conv_2', task_name)
        else:
            conv_2 = self.conv_layer_basic(conv_1_relu, block_name+'/conv_2', task_name)
#        conv_2 = self.conv_layer_basic(conv_1_relu, block_name+'/conv_2', task_name=task_name)
        residual = bottom
        if scale_down:
            residual = tf.nn.avg_pool(residual, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            residual = tf.concat([residual,residual*0.],axis=-1)
        end_relu = tf.nn.relu(conv_2+residual)

        return end_relu, conv_1_relu

    def fc_layer(self, bottom, layer_name, task_name):
        weights, biases = self.get_fc_param(task_name, layer_name)

        fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)

        return fc

    def get_conv_filter_bn(self, task_name, layer_name):
        with tf.variable_scope(task_name+'/'+layer_name):
            weights  = tf.get_variable(name="weights", initializer=self.weight_dict[task_name][layer_name]['weights'], regularizer=self.regularizer_conv)
            beta = tf.constant_initializer(self.weight_dict[task_name][layer_name]['beta'])
            mean = tf.constant_initializer(self.weight_dict[task_name][layer_name]['mean'])
            variance = tf.constant_initializer(self.weight_dict[task_name][layer_name]['variance'])

        return weights , beta, mean, variance

    def get_conv_filter_bn_specific(self, task_name, layer_name):
        with tf.variable_scope(task_name+'/'+layer_name):
            weights  = tf.get_variable(name="weights", initializer=self.weight_dict[task_name][layer_name]['weights'], regularizer=self.regularizer_conv)
            beta = tf.constant_initializer(self.weight_dict[task_name][layer_name]['beta'])
            mean = tf.constant_initializer(self.weight_dict[task_name][layer_name]['mean'])
            variance = tf.constant_initializer(self.weight_dict[task_name][layer_name]['variance'])
            permutation = tf.get_variable(name='permutation', initializer=self.weight_dict[task_name][layer_name]['permutation'], trainable=False)

        return weights , beta, mean, variance, permutation

    def get_conv_filter_shared(self, layer_name):
        with tf.variable_scope('shared/'+layer_name, reuse=tf.AUTO_REUSE):
            weights  = tf.get_variable(name="weights", initializer=self.weight_dict['shared'][layer_name]['weights'], regularizer=self.regularizer_conv)
        return weights

    def get_bn_param(self, task_name, layer_name):
        beta = tf.constant_initializer(self.weight_dict[task_name][layer_name]['beta'])
        mean = tf.constant_initializer(self.weight_dict[task_name][layer_name]['mean'])
        variance = tf.constant_initializer(self.weight_dict[task_name][layer_name]['variance'])

        return beta, mean, variance

    def get_fc_param(self, task_name, layer_name):
        with tf.variable_scope(task_name+'/'+layer_name):
            weights = tf.get_variable(name="weights", initializer=self.weight_dict[task_name][layer_name]['weights'], regularizer=self.regularizer_fc)
            biases = tf.get_variable(name="biases", initializer=self.weight_dict[task_name][layer_name]['biases'], regularizer=self.regularizer_fc)

        return weights, biases

    # load dataset
    def load_dataset(self, task_name):

        def parse_image_train(file_name, label):
            image_string = tf.read_file(file_name)
            image = tf.cast(tf.image.decode_jpeg(image_string), dtype=tf.float32)/255.
            if task_name in ['gtsrb', 'omniglot','svhn']:
                image = tf.image.resize_image_with_crop_or_pad(image, 72,72)
            else:
                image = tf.random_crop(image,size=[64,64,3])
                image = tf.image.random_flip_left_right(image)
            image = (image-means_tensor)/stds_tensor

            label = tf.one_hot(indices=label, depth=n_classes)
            return image, label

        def parse_image_val(file_name, label):
            image_string = tf.read_file(file_name)
            image = tf.cast(tf.image.decode_jpeg(image_string), dtype=tf.float32)/255.
            if task_name in ['gtsrb', 'omniglot','svhn']:
                image = tf.image.resize_image_with_crop_or_pad(image, 72, 72)
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)
            image = (image-means_tensor)/stds_tensor

            label = tf.one_hot(indices=label, depth=n_classes)
            return image, label

        handler=open('decathlon_mean_std.pickle', 'rb')
        dict_mean_std = pickle.load(handler,encoding='bytes')
        means = np.array(dict_mean_std[bytes(task_name + 'mean', encoding='utf8')],dtype=np.float32)
        means_tensor = tf.constant(np.expand_dims(np.expand_dims(means,axis=0),axis=0))
        stds = np.array(dict_mean_std[bytes(task_name + 'std', encoding='utf8')],dtype=np.float32)
        stds_tensor = tf.constant(np.expand_dims(np.expand_dims(stds,axis=0),axis=0))

        imgs_path_train = self.dataset_path+'/'+task_name+'/'+'train/'
        imgs_path_val = self.dataset_path+'/'+task_name+'/'+'val/'
        classes_path_list_train = np.array([imgs_path_train + x +'/' for x in sorted(os.listdir(imgs_path_train))])
        classes_path_list_val = np.array([imgs_path_val + x +'/' for x in sorted(os.listdir(imgs_path_val))])
        n_classes = len(classes_path_list_train)
        self.n_classes[task_name] = n_classes

        filepath_list_train = []
        labels_train = []
        for i in range(len(classes_path_list_train)):
            classes_path = classes_path_list_train[i]
            tmp = [classes_path + x for x in sorted(os.listdir(classes_path))]
            filepath_list_train += tmp
            labels_train += (np.ones(len(tmp))*i).tolist()

        filepath_list_train = np.array(filepath_list_train)
        labels_train = np.array(labels_train,dtype=np.int32)

        self.n_sample_train[task_name] = len(labels_train)
        self.total_batches_train[task_name] = len(labels_train)//self.batch_size+1

        filepath_list_val = []
        labels_val = []
        for i in range(len(classes_path_list_val)):
            classes_path = classes_path_list_val[i]
            tmp = [classes_path + x for x in sorted(os.listdir(classes_path))]
            filepath_list_val += tmp
            labels_val += (np.ones(len(tmp))*i).tolist()

        filepath_list_val = np.array(filepath_list_val)
        labels_val = np.array(labels_val,dtype=np.int32)

        self.n_samples_val[task_name] = len(labels_val)

        file_paths_train = tf.constant(filepath_list_train)
        labels_train = tf.constant(labels_train)

        file_paths_val = tf.constant(filepath_list_val)
        labels_val = tf.constant(labels_val)

        # %% construct input pipeline

        dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        dataset_train = dataset_train.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=100000,count=50))
        dataset_train = dataset_train.map(map_func=parse_image_train, num_parallel_calls=self.cpu_cores)
        dataset_train = dataset_train.batch(self.batch_size)
        dataset_train = dataset_train.prefetch(buffer_size=1)

        dataset_hessian = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        dataset_hessian = dataset_hessian.shuffle(buffer_size=100000)
        dataset_hessian = dataset_hessian.map(map_func=parse_image_val, num_parallel_calls=self.cpu_cores)
        dataset_hessian = dataset_hessian.batch(self.batch_size)
        dataset_hessian = dataset_hessian.prefetch(buffer_size=1)

        dataset_val = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))
        dataset_val = dataset_val.map(map_func=parse_image_val, num_parallel_calls=self.cpu_cores)
        dataset_val = dataset_val.batch(self.batch_size)
        dataset_val = dataset_val.prefetch(buffer_size=1)

        iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        self.X[task_name], self.Y[task_name] = iterator.get_next()

        self.train_init[task_name] = iterator.make_initializer(dataset_train)  # initializer for train_data
        self.test_init[task_name] = iterator.make_initializer(dataset_val)

        self.hessian_init[task_name] = iterator.make_initializer(dataset_hessian)
        # return dataset_train, dataset_val, nb_exp_train, np_exp_val

    def construct_loss(self, task_name):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y[task_name], logits=self.logits[task_name])
        l2_loss = tf.losses.get_regularization_loss(scope=task_name)
        self.loss[task_name] = tf.reduce_mean(entropy, name='loss')+l2_loss

    def construct_optimizer(self, task_name):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        # self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
        #                                       global_step=self.gstep)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.opt[task_name]=tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9, use_nesterov=True)
                self.opt_op[task_name] =self.opt[task_name].minimize(self.loss[task_name])

    def construct_evalutaion(self, task_name):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
#            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(self.logits[task_name], 1), tf.argmax(self.Y[task_name], 1))
            self.accuracy[task_name] = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
#            self.accuracy = tf.reduce_sum(
#                tf.cast(tf.nn.in_top_k(predictions=self.logits, targets=tf.argmax(self.Y, axis=1), k=1),dtype=tf.int32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def joint_finetune(self, steps):
        print('Start finetuning')
        self.sess.run(self.train_init)

        total_loss = 0
        time_last = time.time()

        for i in range(steps):
            _, loss = self.sess.run([self.opt_op, self.loss],feed_dict={'training:0':True}, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

            total_loss += np.sum(list(loss.values()))
#            print('{:d}'.format(i))
            if (i+1) % 5 == 0:
                print('\rbatch={:d}/{:d},curr_loss={:f},used_time:{:.2f}s'.format(i+1,steps,total_loss/i+1,time.time()-time_last),end=' ')
                time_last=time.time()

        print('')



    def train_one_epoch(self, sess, init, epoch, step):
#        start_time = time.time()
        sess.run(init)
        total_loss = 0
        n_batches = 0
        time_last = time.time()

        try:
            while True:
                _, l, summaries = sess.run([self.opt_op, self.loss],feed_dict={'training:0':True})
#                if (step + 1) % self.skip_step == 0:
#                    print('Loss at step {0}: {1}'.format(step+1, l))
                step += 1
                total_loss += l
                n_batches += 1

                if n_batches % 5 ==0:
                            print('\repoch={:d},batch={:d}/{:d},curr_loss={:f},used_time:{:.2f}s'.format(epoch+1,n_batches,self.total_batches_train,total_loss/n_batches,time.time()-time_last),end=' ')
                            time_last=time.time()

        except tf.errors.OutOfRangeError:
            pass
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        sess.run(init)
        total_loss = 0
        total_correct_preds = 0
        n_batches = 0

        try:
            while True:
                loss_batch, accuracy_batch, summaries = sess.run([self.loss, self.accuracy, self.summary_op],feed_dict={'training:0':False})
                writer.add_summary(summaries, global_step=step)
                total_loss += loss_batch
                total_correct_preds += accuracy_batch
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('\nEpoch:{:d}, val_acc={:%}, val_loss={:f}'.format(epoch+1, total_correct_preds / self.n_samples_val, total_loss / n_batches))

    def train(self, n_epochs, lr=None):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        writer = tf.summary.FileWriter('graphs/convnet', tf.get_default_graph())
        if lr is not None:
            self.lr=lr
            self.optimize()

        self.sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.gstep.eval(session=self.sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(self.sess, self.train_init, writer, epoch, step)
            self.eval_once(self.sess, self.test_init, writer, epoch, step)
        writer.close()

    def test_merged(self):
        for task_name in self.merged_task_list:
            self.test(task_name)

    def test(self, task_name):
        self.sess.run(self.test_init[task_name])
        total_loss = 0
        total_correct_preds = 0
        n_batches = 0

        try:
            while True:
                loss_batch, accuracy_batch = self.sess.run([self.loss[task_name], self.accuracy[task_name]],feed_dict={'training:0':False})
                total_loss += loss_batch
                total_correct_preds += accuracy_batch
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('Testing task {:s}, val_acc={:%}, val_loss={:f}'.format(task_name, total_correct_preds / self.n_samples_val[task_name], total_loss / n_batches))

    def fetch_weight_single(self, task_name):
        weight_dict_tensor = dict()
        with tf.variable_scope(task_name+'/pre_conv',reuse=True):
            if self.is_merged_dict[task_name]['pre_conv']:
                weight_dict_tensor['pre_conv'] = {
                        'weights': tf.get_variable('weights'),
                        'beta': tf.get_variable('batch_normalization/beta'),
                        'mean': tf.get_variable('batch_normalization/moving_mean'),
                        'variance': tf.get_variable('batch_normalization/moving_variance'),
                        'permutation': tf.get_variable('permutation', dtype=tf.int32)}
            else:
                weight_dict_tensor['pre_conv'] = {
                        'weights': tf.get_variable('weights'),
                        'beta': tf.get_variable('batch_normalization/beta'),
                        'mean': tf.get_variable('batch_normalization/moving_mean'),
                        'variance': tf.get_variable('batch_normalization/moving_variance')}

        for i in range(self.n_group):
            for j in range(self.n_blocks_per_group):
                block_name = 'conv{:d}_{:d}'.format(i+1,j+1)

                with tf.variable_scope(task_name+'/'+block_name+'/conv_1',reuse=True):
                    if self.is_merged_dict[task_name][block_name+'/conv_1']:
                        weight_dict_tensor[block_name+'/conv_1'] = {
                                'weights': tf.get_variable('weights'),
                                'beta': tf.get_variable('batch_normalization/beta'),
                                'mean': tf.get_variable('batch_normalization/moving_mean'),
                                'variance': tf.get_variable('batch_normalization/moving_variance'),
                                'permutation': tf.get_variable('permutation', dtype=tf.int32)}
                    else:
                        weight_dict_tensor[block_name+'/conv_1'] = {
                                'weights': tf.get_variable('weights'),
                                'beta': tf.get_variable('batch_normalization/beta'),
                                'mean': tf.get_variable('batch_normalization/moving_mean'),
                                'variance': tf.get_variable('batch_normalization/moving_variance')}

                with tf.variable_scope(task_name+'/'+block_name+'/conv_2',reuse=True):
                    if self.is_merged_dict[task_name][block_name+'/conv_2']:
                        weight_dict_tensor[block_name+'/conv_2'] = {
                                'weights': tf.get_variable('weights'),
                                'beta': tf.get_variable('batch_normalization/beta'),
                                'mean': tf.get_variable('batch_normalization/moving_mean'),
                                'variance': tf.get_variable('batch_normalization/moving_variance'),
                                'permutation': tf.get_variable('permutation', dtype=tf.int32)}
                    else:
                        weight_dict_tensor[block_name+'/conv_2'] = {
                                'weights': tf.get_variable('weights'),
                                'beta': tf.get_variable('batch_normalization/beta'),
                                'mean': tf.get_variable('batch_normalization/moving_mean'),
                                'variance': tf.get_variable('batch_normalization/moving_variance')}

        with tf.variable_scope(task_name+'/end_bn',reuse=True):
            weight_dict_tensor['end_bn'] = {
                    'beta': tf.get_variable('batch_normalization/beta'),
                    'mean': tf.get_variable('batch_normalization/moving_mean'),
                    'variance': tf.get_variable('batch_normalization/moving_variance')}

        with tf.variable_scope(task_name+'/classifier',reuse=True):
            weight_dict_tensor['classifier'] = {
                    'weights': tf.get_variable('weights'),
                    'biases': tf.get_variable('biases')}

        return self.sess.run(weight_dict_tensor)

    def fetch_shared_weight(self):
        weight_dict_tensor = dict()
        test_task = self.merged_task_list[0]
        with tf.variable_scope('shared/pre_conv',reuse=True):
            if self.is_merged_dict[test_task]['pre_conv']:
                weight_dict_tensor['pre_conv'] = {'weights': tf.get_variable('weights')}

        for i in range(self.n_group):
            for j in range(self.n_blocks_per_group):
                block_name = 'conv{:d}_{:d}'.format(i+1,j+1)
                with tf.variable_scope('shared/'+block_name+'/conv_1',reuse=True):
                    if self.is_merged_dict[test_task][block_name+'/conv_1']:
                        weight_dict_tensor[block_name+'/conv_1'] = {'weights':tf.get_variable('weights')}

                with tf.variable_scope('shared/'+block_name+'/conv_2',reuse=True):
                    if self.is_merged_dict[test_task][block_name+'/conv_2']:
                        weight_dict_tensor[block_name+'/conv_2'] = {'weights': tf.get_variable('weights'),}

        return self.sess.run(weight_dict_tensor)

    def fetch_weight(self):
        weight_dict = dict()
        task_name_list = list(self.weight_dict.keys())
        for task_name in task_name_list:
            if task_name == 'shared':
                weight_dict['shared'] = self.fetch_shared_weight()
            else:
                weight_dict[task_name] = self.fetch_weight_single(task_name)

        return weight_dict

    def save_weight(self, save_path):
        self.weight_dict=self.fetch_weight()
        filehandler = open(save_path, 'wb')
        pickle.dump((self.weight_dict, self.is_merged_dict, self.merged_task_list),filehandler)
        filehandler.close()

#%%

if __name__ == '__main__':
    model = ResNet_MTZ(dataset_path='/srv/node/sdc1/image_data/decathlon/', model_path='model_weights/shared_gtsrb_ucf101')
    model.build()
    model.test_merged()
#    model.add_new_task('gtsrb')
#    model.add_new_task('ucf101')
#    model.build()
#    model.test('gtsrb')
#    model.test('ucf101')
#    model.merge_initial_tasks('gtsrb','ucf101')
#    model.save_weight('model_weights/shared_gtsrb_ucf101')
#
##    'model_weights/shared_gtsrb_ucf101_)
    task_list = ['svhn','omniglot','cifar100','aircraft','dtd','vgg-flowers']
    save_path = 'model_weights/shared_gtsrb_ucf101'
    for task_name in task_list:
        print('==================================')
        model = ResNet_MTZ(model_path=save_path)
        model.add_new_task(task_name)
        model.build()
        model.test_merged()
        model.test(task_name)
        model.merge_new_task(task_name)
        save_path = save_path + '_' + task_name
        model.save_weight(save_path)
