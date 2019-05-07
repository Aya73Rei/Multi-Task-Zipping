#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 02:09:01 2018

@author: hex
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
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

from .models_merged_construct_dropout import VGG_merged
from .dataset_construct_random_crop import load_celebA, load_imageNet
from sys import getsizeof

tf.reset_default_graph()

# %% hyper params
BATCH_SIZE = 32

EVAL_BATCHES = 300
HESSIAN_BATCH = 200
FINETUNE_BATCHES = 500

CPU_CORES = 8

require_improvement = 2

train_ratio_imageNet2Face = 1

save_dir = './Experiment_1_Final/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# %% functions
def evaluate(iterator, dataset, accuracy_sum, nb_exp_val, nb_batches):
    print('start evaluating')
    if nb_batches > nb_exp_val // BATCH_SIZE:
        nb_exp = nb_exp_val
        nb_batches = nb_exp_val // BATCH_SIZE
    else:
        nb_exp = nb_batches * BATCH_SIZE
    sess.run(iterator.make_initializer(dataset))
    acc_sum = 0
    batch_count = 0
    curr_time = time.time()
    for i in range(nb_batches):
        try:
            batch_count += 1
            acc_sum += sess.run(accuracy_sum, feed_dict={keep_ratio: 1})
            if batch_count % 10 == 0:
                print('\rbatch={:d}/{:d},used_time:{:.2f}s'.format(batch_count, nb_batches, time.time() - curr_time),
                      end=' ')
                curr_time = time.time()
        except tf.errors.OutOfRangeError:
            break
    print('\n')

    return acc_sum / nb_exp


def calculate_hessian_conv_tf(layer_inputs):
    a = tf.expand_dims(layer_inputs, axis=-1)
    a = tf.concat([a, tf.ones([tf.shape(a)[0], tf.shape(a)[1], tf.shape(a)[2], 1, 1])], axis=3)
    b = tf.expand_dims(layer_inputs, axis=3)
    b = tf.concat([b, tf.ones([tf.shape(b)[0], tf.shape(b)[1], tf.shape(b)[2], 1, 1])], axis=4)
    # print 'b shape: %s' %b.get_shape()
    outprod = tf.multiply(a, b)
    # print 'outprod shape: %s' %outprod.get_shape()
    return tf.reduce_mean(outprod, axis=[0, 1, 2])


def calculate_hessian_fc_tf(layer_inputs):
    a = tf.expand_dims(layer_inputs, axis=-1)
    a = tf.concat([a, tf.ones([tf.shape(a)[0], 1, 1])], axis=1)
    b = tf.expand_dims(layer_inputs, axis=1)
    b = tf.concat([b, tf.ones([tf.shape(b)[0], 1, 1])], axis=2)
    outprod = tf.multiply(a, b)

    return tf.reduce_mean(outprod, axis=0)


def run_hessian(op, iterator, dataset, nb_exp, nb_batches, BATCH_SIZE_HESSIAN):
    print('start calculating hessian')
    if nb_batches > nb_exp // BATCH_SIZE_HESSIAN:
        nb_batches = nb_exp // BATCH_SIZE_HESSIAN
    sess.run(iterator.make_initializer(dataset))
    curr_time = time.time()
    for batch_count in range(nb_batches):
        if batch_count == 0:
            hessian_sum = sess.run(op, feed_dict={keep_ratio: 1})
        else:
            hessian_sum += sess.run(op, feed_dict={keep_ratio: 1})
        if (batch_count + 1) % 10 == 0:
            print('\rbatch={:d}/{:d},used_time:{:.2f}s'.format(batch_count + 1, nb_batches, time.time() - curr_time),
                  end=' ')
            curr_time = time.time()
    print('\n')
    return hessian_sum / nb_batches


def fine_tune(nb_batches, trian_keep_ratio):
    if nb_batches > (nb_exp_imageNet_train * 4) // BATCH_SIZE:
        nb_batches = (nb_exp_imageNet_train * 4) // BATCH_SIZE
    print('start fine tuning')
    sess.run(iterator_Face.make_initializer(dataset_Face_train))
    sess.run(iterator_imageNet.make_initializer(dataset_imageNet_train))
    curr_time = time.time()
    batch_count = 0
    for batch_count in range(nb_batches):
        try:
            #        sess.run(opt_op_imageNet, feed_dict={keep_ratio:0.5})
            #        sess.run(opt_op_Face, feed_dict={keep_ratio:0.5})
            sess.run(opt_op, feed_dict={keep_ratio: trian_keep_ratio})

            #        print('loss={:f}'.format(sess.run(loss_imageNet,feed_dict={keep_ratio:0.5})))
            #        print('loss={:f}'.format(sess.run(loss_Face,feed_dict={keep_ratio:0.5})))
            batch_count += 1
            if batch_count % 10 == 0:
                loss_Face_vl = sess.run(loss_Face, feed_dict={keep_ratio: 1})
                loss_imageNet_vl = sess.run(loss_imageNet, feed_dict={keep_ratio: 1})
                print('\rbatch={:d}/{:d},loss_Face={:f},loss_imageNet={:f},used_time:{:.2f}s'.format(batch_count,
                                                                                                     nb_batches,
                                                                                                     loss_Face_vl,
                                                                                                     loss_imageNet_vl,
                                                                                                     time.time() - curr_time),
                      end=' ')
                #                print('\rbatch={:d}/{:d},used_time:{:.2f}s'.format(batch_count, nb_batches, time.time()-curr_time),end=' ')
                curr_time = time.time()
        except tf.errors.OutOfRangeError:
            break
    print('\n')


def fine_tune_imageNet(nb_batches):
    if nb_batches > (nb_exp_imageNet_train * 4) // BATCH_SIZE:
        nb_batches = (nb_exp_imageNet_train * 4) // BATCH_SIZE
    print('start fine tuning')
    sess.run(iterator_Face.make_initializer(dataset_Face_train))
    sess.run(iterator_imageNet.make_initializer(dataset_imageNet_train))
    curr_time = time.time()
    batch_count = 0
    for batch_count in range(nb_batches):
        sess.run(opt_op_imageNet, feed_dict={keep_ratio: 0.5})

        #        print('loss={:f}'.format(sess.run(loss_imageNet,feed_dict={keep_ratio:0.5})))

        batch_count += 1
        if batch_count % 10 == 0:
            loss_Face_vl = sess.run(loss_Face, feed_dict={keep_ratio: 1})
            loss_imageNet_vl = sess.run(loss_imageNet, feed_dict={keep_ratio: 1})
            print(
                '\rbatch={:d}/{:d},loss_Face={:f},loss_imageNet={:f},used_time:{:.2f}s'.format(batch_count, nb_batches,
                                                                                               loss_Face_vl,
                                                                                               loss_imageNet_vl,
                                                                                               time.time() - curr_time),
                end=' ')
            #            print('\rbatch={:d}/{:d},used_time:{:.2f}s'.format(batch_count, nb_batches, time.time()-curr_time),end=' ')
            curr_time = time.time()
    print('\n')


def generate_merge_list_conv():
    input_shape = np.asarray(layer_inputs_Face[layer_name].get_shape().as_list()[1:])

    BATCH_SIZE_HESSIAN = (224 * 224 * 64 * 64 * 2) / (input_shape[0] * input_shape[1] * input_shape[2] * input_shape[2])
    BATCH_SIZE_HESSIAN = int(BATCH_SIZE_HESSIAN)

    dataset_Face_train, dataset_Face_val, nb_exp_Face_train, nb_exp_Face_val = load_celebA(BATCH_SIZE_HESSIAN,
                                                                                           CPU_CORES)

    dataset_imageNet_train, dataset_imageNet_val, nb_exp_imageNet_train, nb_exp_imageNet_val = load_imageNet(
        BATCH_SIZE_HESSIAN, CPU_CORES)

    nb_batches_hessian = (HESSIAN_BATCH * BATCH_SIZE) // BATCH_SIZE_HESSIAN + 1

    hessian_op_A = calculate_hessian_conv_tf(layer_inputs_Face[layer_name])
    hessian_inverse_A = run_hessian(hessian_op_A, iterator_Face, dataset_Face_train, nb_exp_Face_train,
                                    nb_batches_hessian, BATCH_SIZE_HESSIAN)
    hessian_inverse_A = np.linalg.inv(hessian_inverse_A)

    hessian_op_B = calculate_hessian_conv_tf(layer_inputs_imageNet[layer_name])
    hessian_inverse_B = run_hessian(hessian_op_B, iterator_imageNet, dataset_imageNet_train, nb_exp_imageNet_train,
                                    nb_batches_hessian, BATCH_SIZE_HESSIAN)
    hessian_inverse_B = np.linalg.inv(hessian_inverse_B)

    H_hat = np.linalg.inv(hessian_inverse_A + hessian_inverse_B)
    MtxA = np.dot(hessian_inverse_A, H_hat)

    err_mtx = np.zeros((kernel_nb, kernel_nb, kernel_size, kernel_size))

    b_rep_A = np.tile(bias_VGG_Face[layer_name], (kernel_size, kernel_size, 1, 1))
    Wb_A = np.concatenate((Weights_VGG_Face[layer_name][:, :, :last_kernel_nb, :], b_rep_A), axis=2)

    b_rep_B = np.tile(bias_imageNet[layer_name], (kernel_size, kernel_size, 1, 1))
    Wb_B = np.concatenate((Weights_imageNet[layer_name][:, :, :last_kernel_nb, :], b_rep_B), axis=2)

    merged_vector = np.zeros((kernel_size, kernel_size, Wb_A.shape[2], kernel_nb, kernel_nb))

    for i in range(kernel_nb):
        #        print('i={:d}'.format(i))
        for j in range(kernel_nb):
            for kernel_i in range(kernel_size):
                for kernel_j in range(kernel_size):
                    dist_vec = Wb_A[kernel_i, kernel_j, :, i] - Wb_B[kernel_i, kernel_j, :, j]
                    delta_A = np.dot(MtxA, -dist_vec)
                    merged_vector[kernel_i, kernel_j, :, i, j] = Wb_A[kernel_i, kernel_j, :, i] + delta_A
                    #                    merged_vector[kernel_i,kernel_j,:,i,j]=(Wb_A[kernel_i,kernel_j,:,i]+Wb_B[kernel_i,kernel_j,:,j])/2
                    err_mtx[i, j, kernel_i, kernel_j] = 0.5 * np.dot(np.dot(dist_vec, H_hat), dist_vec)

    err_mtx = np.sum(err_mtx, axis=(2, 3))

    plt.hist(np.nan_to_num(err_mtx.flatten()), bins=500)
    plt.show()

    sorted_index = np.argsort(err_mtx, axis=None)

    merge_list_all = []
    merge_loss = []
    count = 0
    i = 0

    for i in range(kernel_nb ** 2):
        l_index = sorted_index[i] // kernel_nb
        r_index = sorted_index[i] % kernel_nb
        if count == 0:
            merge_list_all.append((l_index, r_index))
            merge_loss.append(err_mtx[l_index, r_index])
            count += 1
        elif np.all(np.asarray(merge_list_all)[:, 0] != l_index) and np.all(
                np.asarray(merge_list_all)[:, 1] != r_index):
            merge_list_all.append((l_index, r_index))
            merge_loss.append(err_mtx[l_index, r_index])
            count += 1
        #       print('i={:d}'.format(i))

    plt.plot(merge_loss)
    plt.show()

    return np.asarray(merge_list_all), merged_vector, err_mtx, merge_loss


def generate_merge_list_fc():
    #    input_shape=sess.run(tf.shape(layer_inputs_Face[layer_name]))

    BATCH_SIZE_HESSIAN = 32
    BATCH_SIZE_HESSIAN = int(BATCH_SIZE_HESSIAN)

    dataset_Face_train, dataset_Face_val, nb_exp_Face_train, nb_exp_Face_val = load_celebA(BATCH_SIZE_HESSIAN,
                                                                                           CPU_CORES)

    dataset_imageNet_train, dataset_imageNet_val, nb_exp_imageNet_train, nb_exp_imageNet_val = load_imageNet(
        BATCH_SIZE_HESSIAN, CPU_CORES)

    nb_batches_hessian = (HESSIAN_BATCH * BATCH_SIZE) // BATCH_SIZE_HESSIAN + 1

    hessian_op_A = calculate_hessian_fc_tf(layer_inputs_Face[layer_name])
    hessian_A = run_hessian(hessian_op_A, iterator_Face, dataset_Face_train, nb_exp_Face_train, nb_batches_hessian,
                            BATCH_SIZE_HESSIAN)
    hessian_inverse_A = np.linalg.pinv(hessian_A)

    hessian_op_B = calculate_hessian_fc_tf(layer_inputs_imageNet[layer_name])
    hessian_B = run_hessian(hessian_op_B, iterator_imageNet, dataset_imageNet_train, nb_exp_imageNet_train,
                            nb_batches_hessian, BATCH_SIZE_HESSIAN)
    hessian_inverse_B = np.linalg.pinv(hessian_B)

    H_hat = np.linalg.pinv(hessian_inverse_A + hessian_inverse_B)
    MtxA = np.dot(hessian_inverse_A, H_hat)
    #    MtxB=np.dot(hessian_inverse_B,H_hat)

    err_mtx = np.zeros((kernel_nb, kernel_nb))

    b_rep_A = np.expand_dims(bias_VGG_Face[layer_name], axis=0)
    Wb_A = np.concatenate((Weights_VGG_Face[layer_name][:last_kernel_nb, :], b_rep_A), axis=0)

    b_rep_B = np.expand_dims(bias_imageNet[layer_name], axis=0)
    Wb_B = np.concatenate((Weights_imageNet[layer_name][:last_kernel_nb, :], b_rep_B), axis=0)

    #    merged_vector=np.zeros((Wb_A.shape[0],kernel_nb,kernel_nb))

    #    curr_time=time.time()
    for i in range(kernel_nb):
        for j in range(kernel_nb):
            #            dist_vec=Wb_A[:,i]-Wb_B[:,j]
            #            delta_A=np.dot(MtxA,-dist_vec)
            #            delta_B=np.dot(MtxB,dist_vec)
            #            merged_vector[:,i,j]=Wb_A[:,i]+delta_A
            #            err_mtx[i,j]=0.5*np.dot(np.dot(dist_vec,H_hat),dist_vec)
            err_mtx[i, j] = np.linalg.norm(Wb_A[:, i] - Wb_B[:, j])

        print('\ri={:d}'.format(i), end=' ')
    #        curr_time=time.time()

    plt.hist(err_mtx.flatten(), bins=500)
    plt.show()

    sorted_index = np.argsort(err_mtx, axis=None)

    merge_list_all = np.ones((kernel_nb, 2), dtype=np.int16) * kernel_nb + 1
    merge_loss = np.zeros(kernel_nb)
    merged_vector = np.zeros((kernel_nb, last_kernel_nb + 1))
    count = 0
    i = 0

    for i in range(kernel_nb ** 2):
        l_index = sorted_index[i] // kernel_nb
        r_index = sorted_index[i] % kernel_nb
        if count == 0:
            merge_list_all[count] = [l_index, r_index]
            merge_loss[count] = err_mtx[l_index, r_index]
            merged_vector[count, :] = Wb_A[:, l_index] + np.dot(MtxA, Wb_B[:, r_index] - Wb_A[:, l_index])
            count += 1
        #           print('\rfound {:d}'.format(count),end=' ')
        elif np.all(merge_list_all[:, 0] != l_index) and np.all(merge_list_all[:, 1] != r_index):
            merge_list_all[count] = [l_index, r_index]
            merge_loss[count] = err_mtx[l_index, r_index]
            merged_vector[count, :] = Wb_A[:, l_index] + np.dot(MtxA, Wb_B[:, r_index] - Wb_A[:, l_index])
            count += 1
        #           print('\rfound {:d}'.format(count),end=' ')
        if i % 10000 == 0:
            print('\ri={:d}/{:d},count={:d}'.format(i, kernel_nb ** 2, count), end=' ')

    plt.plot(merge_loss)
    plt.show()

    return merge_list_all, merged_vector, err_mtx, merge_loss


def generate_merge_list_fc6():
    err_mtx = np.zeros((kernel_nb, kernel_nb))

    b_rep_A = np.expand_dims(bias_VGG_Face[layer_name], axis=0)
    Wb_A = np.concatenate(
        (Weights_VGG_Face[layer_name].reshape(7, 7, 512, -1)[:, :, :last_kernel_nb, :].reshape(-1, 4096), b_rep_A),
        axis=0)

    b_rep_B = np.expand_dims(bias_imageNet[layer_name], axis=0)
    Wb_B = np.concatenate(
        (Weights_imageNet[layer_name].reshape(7, 7, 512, -1)[:, :, :last_kernel_nb, :].reshape(-1, 4096), b_rep_B),
        axis=0)

    #    merged_vector=np.zeros((Wb_A.shape[0],kernel_nb,kernel_nb))

    for i in range(kernel_nb):
        print('\ri={:d}'.format(i), end=' ')
        for j in range(kernel_nb):
            err_mtx[i, j] = np.linalg.norm(Wb_A[:, i] - Wb_B[:, j])

    print(' ')
    plt.hist(err_mtx.flatten(), bins=500)
    plt.show()

    sorted_index = np.argsort(err_mtx, axis=None)

    merge_list_all = np.ones((kernel_nb, 2), dtype=np.int16) * kernel_nb + 1
    merge_loss = np.zeros(kernel_nb)
    merged_vector = np.zeros((kernel_nb, last_kernel_nb * 7 * 7 + 1))
    count = 0
    i = 0

    for i in range(kernel_nb ** 2):
        l_index = sorted_index[i] // kernel_nb
        r_index = sorted_index[i] % kernel_nb
        if count == 0:
            merge_list_all[count] = [l_index, r_index]
            merge_loss[count] = err_mtx[l_index, r_index]
            merged_vector[count, :] = (Wb_A[:, l_index] + Wb_B[:, r_index]) / 2
            count += 1
        elif np.all(np.asarray(merge_list_all)[:, 0] != l_index) and np.all(
                np.asarray(merge_list_all)[:, 1] != r_index):
            merge_list_all[count] = [l_index, r_index]
            merge_loss[count] = err_mtx[l_index, r_index]
            merged_vector[count, :] = (Wb_A[:, l_index] + Wb_B[:, r_index]) / 2
            count += 1
        #       print('i={:d}'.format(i))
    plt.plot(merge_loss)
    plt.show()

    return merge_list_all, merged_vector, err_mtx, merge_loss


# %%
merged_nb_dict = {
    'conv1_1': 0,
    'conv1_2': 0,
    'conv2_1': 0,
    'conv2_2': 0,
    'conv3_1': 0,
    'conv3_2': 0,
    'conv3_3': 0,
    'conv4_1': 0,
    'conv4_2': 0,
    'conv4_3': 0,
    'conv5_1': 0,
    'conv5_2': 0,
    'conv5_3': 0,
    'fc6': 0,
    'fc7': 0,
}

original_size_dict = {
    'conv1_1': 64,
    'conv1_2': 64,
    'conv2_1': 128,
    'conv2_2': 128,
    'conv3_1': 256,
    'conv3_2': 256,
    'conv3_3': 256,
    'conv4_1': 512,
    'conv4_2': 512,
    'conv4_3': 512,
    'conv5_1': 512,
    'conv5_2': 512,
    'conv5_3': 512,
    'fc6': 4096,
    'fc7': 4096,
}

layer_name_list = ['conv1_1',
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
# %%
training_iter_dict = {
    'conv1_1': 50,
    'conv1_2': 100,
    'conv2_1': 100,
    'conv2_2': 100,
    'conv3_1': 100,
    'conv3_2': 100,
    'conv3_3': 100,
    'conv4_1': 400,
    'conv4_2': 400,
    'conv4_3': 400,
    'conv5_1': 400,
    'conv5_2': 400,
    'conv5_3': 1000,
    'fc6': 2000,
    'fc7': 5000,
}

# %%
tomerge_nb_dict = {
    'conv1_1': 64,
    'conv1_2': 64,
    'conv2_1': 128,
    'conv2_2': 128,
    'conv3_1': 256,
    'conv3_2': 256,
    'conv3_3': 256,
    'conv4_1': 512,
    'conv4_2': 512,
    'conv4_3': 512,
    'conv5_1': 512,
    'conv5_2': 512,
    'conv5_3': 512,
    'fc6': 4096,
    'fc7': 4096,
}

# %% load dataset

dataset_Face_train, dataset_Face_val, nb_exp_Face_train, nb_exp_Face_val = load_celebA(BATCH_SIZE, CPU_CORES)

dataset_imageNet_train, dataset_imageNet_val, nb_exp_imageNet_train, nb_exp_imageNet_val = load_imageNet(BATCH_SIZE,
                                                                                                         CPU_CORES)

iterator_Face = tf.data.Iterator.from_structure(dataset_Face_train.output_types, dataset_Face_train.output_shapes)

x_Face, y_true_Face = iterator_Face.get_next()

iterator_imageNet = tf.data.Iterator.from_structure(dataset_imageNet_train.output_types,
                                                    dataset_imageNet_train.output_shapes)

x_imageNet, y_true_imageNet = iterator_imageNet.get_next()

# %% construct model
keep_ratio = tf.placeholder(tf.float32)
y_Face, y_imageNet, W_merged, b_merged, W_inter_Face, W_VGG_Face, b_VGG_Face, layer_inputs_Face, W_inter_imageNet, W_imageNet, b_imageNet, layer_inputs_imageNet = VGG_merged(
    x_Face, x_imageNet, merged_nb_dict, keep_ratio)

accuracy_sum_Face = tf.reduce_sum(tf.cast(tf.equal(y_true_Face, tf.round(tf.sigmoid(y_Face))), tf.float32)) / 40
# top1_sum_imageNet = tf.reduce_sum(tf.cast(tf.nn.in_top_k(predictions=y_imageNet, targets=y_true_imageNet, k=1),dtype=tf.int32))
accuracy_sum_imageNet = tf.reduce_sum(
    tf.cast(tf.nn.in_top_k(predictions=y_imageNet, targets=tf.argmax(y_true_imageNet, axis=1), k=5), dtype=tf.int32))

#    opt_op_Face=opt_Face.minimize(loss_Face)
#    opt_op_imageNet=opt_imageNet.minimize(loss_imageNet)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# %% load weights
Weights_VGG_Face, bias_VGG_Face = pickle.load(open('tf_saved_VGG_face/unpruned_weights_phase2', 'rb'))
Weights_imageNet, bias_imageNet = pickle.load(open('Weights_imageNet', 'rb'))

for layer_name in list(W_VGG_Face.keys()):
    sess.run(tf.assign(W_VGG_Face[layer_name], Weights_VGG_Face[layer_name]))
    sess.run(tf.assign(b_VGG_Face[layer_name], bias_VGG_Face[layer_name]))
    sess.run(tf.assign(W_imageNet[layer_name], Weights_imageNet[layer_name]))
    sess.run(tf.assign(b_imageNet[layer_name], bias_imageNet[layer_name]))
# for layer_name in list(Weights_merged.keys()):
#    sess.run(tf.assign(W_merged[layer_name],Weights_merged[layer_name]))
#    sess.run(tf.assign(b_merged[layer_name],bias_merged[layer_name]))
# for layer_name in list(Weights_inter_Face.keys()):
#    sess.run(tf.assign(W_inter_Face[layer_name],Weights_inter_Face[layer_name]))
#    sess.run(tf.assign(W_inter_imageNet[layer_name],Weights_inter_imageNet[layer_name]))
# %%
# saver = tf.train.Saver()
# save_dir = 'tf_saved_VGG_merge/'
# if not os.path.exists(save_dir):
#    os.makedirs(save_dir)
# save_path = os.path.join(save_dir, 'best_validation')
##saver.save(sess=sess, save_path=save_path)
# summary_writer = tf.summary.FileWriter(save_dir+'test')
# summary_writer.add_graph(sess.graph)

# %% test acc
acc_Face = []
acc_imageNet = []

acc_Face_initial = evaluate(iterator_Face, dataset_Face_val, accuracy_sum_Face, nb_exp_Face_val, EVAL_BATCHES)
acc_imageNet_initial = evaluate(iterator_imageNet, dataset_imageNet_val, accuracy_sum_imageNet, nb_exp_imageNet_val,
                                EVAL_BATCHES)

acc_Face.append(acc_Face_initial)
acc_imageNet.append(acc_imageNet_initial)

print('Initial: Acc_Face={:%}, Acc_imageNet={:%}'.format(acc_Face_initial, acc_imageNet_initial))

# %%
Weights_merged = dict()
bias_merged = dict()
Weights_inter_Face = dict()
Weights_inter_imageNet = dict()
merged_layer_name_list = []

# %%
for layer_index in range(15):
    pass
    # %%
    layer_name = layer_name_list[layer_index]
    print('merging layer' + layer_name)
    # %%
    if layer_index > 0:
        (Weights_merged, Weights_VGG_Face, Weights_imageNet, Weights_inter_Face, Weights_inter_imageNet, bias_merged,
         bias_VGG_Face, bias_imageNet, merged_nb_dict, _, _, _) = pickle.load(open(
            save_dir + layer_name_list[layer_index - 1] + '_{:d}_merged_finetuned'.format(
                merged_nb_dict[layer_name_list[layer_index - 1]]), 'rb'))

    # %% calculate hessian
    #    sess.run(iterator_Face.make_initializer(dataset_Face_train))

    # %%
    if layer_index < 13:

        if layer_index == 0:
            last_kernel_nb = 3
        else:
            last_kernel_nb = merged_nb_dict[layer_name_list[layer_index - 1]]

        kernel_size = Weights_VGG_Face[layer_name].shape[0]
        kernel_nb = Weights_VGG_Face[layer_name].shape[3]

        merge_list_all, merged_vector, err_mtx, merge_loss = generate_merge_list_conv()
        # %
        tomerge_nb = tomerge_nb_dict[layer_name]
        merge_list = merge_list_all[:tomerge_nb]

        mask_Face = np.ones(kernel_nb, dtype=bool)
        mask_imageNet = np.ones(kernel_nb, dtype=bool)
        mask_Face[merge_list[:, 0]] = False
        mask_imageNet[merge_list[:, 1]] = False

        Weights_merged[layer_name] = merged_vector[:, :, :-1, merge_list[:, 0], merge_list[:, 1]]
        bias_merged[layer_name] = np.mean(merged_vector[:, :, -1, merge_list[:, 0], merge_list[:, 1]], axis=(0, 1))

        Weights_inter_Face[layer_name] = Weights_VGG_Face[layer_name][:, :, last_kernel_nb:, merge_list[:, 0]]
        Weights_inter_imageNet[layer_name] = Weights_imageNet[layer_name][:, :, last_kernel_nb:, merge_list[:, 1]]

        Weights_VGG_Face[layer_name] = Weights_VGG_Face[layer_name][:, :, :, mask_Face]
        bias_VGG_Face[layer_name] = bias_VGG_Face[layer_name][mask_Face]

        Weights_imageNet[layer_name] = Weights_imageNet[layer_name][:, :, :, mask_imageNet]
        bias_imageNet[layer_name] = bias_imageNet[layer_name][mask_imageNet]

        index_Face = np.concatenate((merge_list[:, 0], np.arange(kernel_nb)[mask_Face]))
        index_imageNet = np.concatenate((merge_list[:, 1], np.arange(kernel_nb)[mask_imageNet]))

        if layer_name == 'conv5_3':
            Weights_VGG_Face[layer_name_list[layer_index + 1]] = Weights_VGG_Face[
                                                                     layer_name_list[layer_index + 1]].reshape(7, 7,
                                                                                                               512, -1)[
                                                                 :, :, index_Face, :].reshape(-1, 4096)
            Weights_imageNet[layer_name_list[layer_index + 1]] = Weights_imageNet[
                                                                     layer_name_list[layer_index + 1]].reshape(7, 7,
                                                                                                               512, -1)[
                                                                 :, :, index_imageNet, :].reshape(-1, 4096)
        else:
            Weights_VGG_Face[layer_name_list[layer_index + 1]] = Weights_VGG_Face[layer_name_list[layer_index + 1]][:,
                                                                 :, index_Face, :]
            Weights_imageNet[layer_name_list[layer_index + 1]] = Weights_imageNet[layer_name_list[layer_index + 1]][:,
                                                                 :, index_imageNet, :]

    elif layer_name == 'fc6':
        last_kernel_nb = merged_nb_dict[layer_name_list[layer_index - 1]]
        kernel_nb = Weights_VGG_Face[layer_name].shape[-1]

        merge_list_all, merged_vector, err_mtx, merge_loss = generate_merge_list_fc6()
        np.save(save_dir + 'merge_list_all_fc6', merge_list_all)
        np.save(save_dir + 'merged_vector_fc6', merged_vector)
        np.save(save_dir + 'err_mtx_fc6', err_mtx)
        np.save(save_dir + 'merge_loss_fc6', merge_loss)

        tomerge_nb = tomerge_nb_dict[layer_name]
        merge_list = merge_list_all[:tomerge_nb]

        mask_Face = np.ones(kernel_nb, dtype=bool)
        mask_imageNet = np.ones(kernel_nb, dtype=bool)
        mask_Face[merge_list[:, 0]] = False
        mask_imageNet[merge_list[:, 1]] = False

        Weights_merged[layer_name] = merged_vector[:tomerge_nb, :-1].transpose()
        bias_merged[layer_name] = merged_vector[:tomerge_nb, -1]

        Weights_inter_Face[layer_name] = Weights_VGG_Face[layer_name].reshape(7, 7, 512, -1)[:, :, last_kernel_nb:,
                                         merge_list[:, 0]].reshape(-1, tomerge_nb)
        Weights_inter_imageNet[layer_name] = Weights_imageNet[layer_name].reshape(7, 7, 512, -1)[:, :, last_kernel_nb:,
                                             merge_list[:, 1]].reshape(-1, tomerge_nb)

        Weights_VGG_Face[layer_name] = Weights_VGG_Face[layer_name][:, mask_Face]
        bias_VGG_Face[layer_name] = bias_VGG_Face[layer_name][mask_Face]

        Weights_imageNet[layer_name] = Weights_imageNet[layer_name][:, mask_imageNet]
        bias_imageNet[layer_name] = bias_imageNet[layer_name][mask_imageNet]

        index_Face = np.concatenate((merge_list[:, 0], np.arange(kernel_nb)[mask_Face]))
        index_imageNet = np.concatenate((merge_list[:, 1], np.arange(kernel_nb)[mask_imageNet]))

        Weights_VGG_Face[layer_name_list[layer_index + 1]] = Weights_VGG_Face[layer_name_list[layer_index + 1]][
                                                             index_Face, :]
        Weights_imageNet[layer_name_list[layer_index + 1]] = Weights_imageNet[layer_name_list[layer_index + 1]][
                                                             index_imageNet, :]

    else:
        last_kernel_nb = merged_nb_dict[layer_name_list[layer_index - 1]]
        kernel_nb = Weights_VGG_Face[layer_name].shape[-1]
        merge_list_all, merged_vector, err_mtx, merge_loss = generate_merge_list_fc()

        np.save(save_dir + 'merge_list_all_' + layer_name, merge_list_all)
        np.save(save_dir + 'merged_vector_' + layer_name, merged_vector)
        np.save(save_dir + 'err_mtx_' + layer_name, err_mtx)
        np.save(save_dir + 'merge_loss_fc6', merge_loss)

        tomerge_nb = tomerge_nb_dict[layer_name]
        merge_list = merge_list_all[:tomerge_nb]

        mask_Face = np.ones(kernel_nb, dtype=bool)
        mask_imageNet = np.ones(kernel_nb, dtype=bool)
        mask_Face[merge_list[:, 0]] = False
        mask_imageNet[merge_list[:, 1]] = False

        Weights_merged[layer_name] = merged_vector[:tomerge_nb, :-1].transpose()
        bias_merged[layer_name] = merged_vector[:tomerge_nb, -1]

        Weights_inter_Face[layer_name] = Weights_VGG_Face[layer_name][last_kernel_nb:, merge_list[:, 0]]
        Weights_inter_imageNet[layer_name] = Weights_imageNet[layer_name][last_kernel_nb:, merge_list[:, 1]]

        Weights_VGG_Face[layer_name] = Weights_VGG_Face[layer_name][:, mask_Face]
        bias_VGG_Face[layer_name] = bias_VGG_Face[layer_name][mask_Face]

        Weights_imageNet[layer_name] = Weights_imageNet[layer_name][:, mask_imageNet]
        bias_imageNet[layer_name] = bias_imageNet[layer_name][mask_imageNet]

        index_Face = np.concatenate((merge_list[:, 0], np.arange(kernel_nb)[mask_Face]))
        index_imageNet = np.concatenate((merge_list[:, 1], np.arange(kernel_nb)[mask_imageNet]))

        Weights_VGG_Face[layer_name_list[layer_index + 1]] = Weights_VGG_Face[layer_name_list[layer_index + 1]][
                                                             index_Face, :]
        Weights_imageNet[layer_name_list[layer_index + 1]] = Weights_imageNet[layer_name_list[layer_index + 1]][
                                                             index_imageNet, :]

    merged_nb_dict[layer_name] = tomerge_nb_dict[layer_name]
    # %%
    tf.reset_default_graph()

    # % load dataset
    dataset_Face_train, dataset_Face_val, nb_exp_Face_train, nb_exp_Face_val = load_celebA(BATCH_SIZE, CPU_CORES)

    dataset_imageNet_train, dataset_imageNet_val, nb_exp_imageNet_train, nb_exp_imageNet_val = load_imageNet(BATCH_SIZE,
                                                                                                             CPU_CORES)

    iterator_Face = tf.data.Iterator.from_structure(dataset_Face_train.output_types, dataset_Face_train.output_shapes)

    x_Face, y_true_Face = iterator_Face.get_next()

    iterator_imageNet = tf.data.Iterator.from_structure(dataset_imageNet_train.output_types,
                                                        dataset_imageNet_train.output_shapes)

    x_imageNet, y_true_imageNet = iterator_imageNet.get_next()

    # % construct model
    keep_ratio = tf.placeholder(tf.float32)

    y_Face, y_imageNet, W_merged, b_merged, W_inter_Face, W_VGG_Face, b_VGG_Face, layer_inputs_Face, W_inter_imageNet, W_imageNet, b_imageNet, layer_inputs_imageNet = VGG_merged(
        x_Face, x_imageNet, merged_nb_dict, keep_ratio)

    loss_Face = tf.losses.mean_squared_error(labels=y_true_Face, predictions=tf.nn.sigmoid(y_Face))
    #    loss_imageNet = tf.losses.mean_squared_error(labels=y_true_imageNet, predictions=tf.nn.softmax(y_imageNet))
    loss_imageNet = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true_imageNet, logits=y_imageNet))

    accuracy_sum_Face = tf.reduce_sum(tf.cast(tf.equal(y_true_Face, tf.round(tf.sigmoid(y_Face))), tf.float32)) / 40
    # top1_sum_imageNet = tf.reduce_sum(tf.cast(tf.nn.in_top_k(predictions=y_imageNet, targets=y_true_imageNet, k=1),dtype=tf.int32))
    accuracy_sum_imageNet = tf.reduce_sum(
        tf.cast(tf.nn.in_top_k(predictions=y_imageNet, targets=tf.argmax(y_true_imageNet, axis=1), k=5),
                dtype=tf.int32))

    #    opt_Face = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9, use_nesterov=True)
    #    opt_imageNet = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9, use_nesterov=True)

    #    gradients_Face=opt_Face.compute_gradients(loss_Face)
    #    gradients_imageNet=opt_imageNet.compute_gradients(loss_imageNet)

    #    opt_op_Face=opt_Face.apply_gradients(gradients_Face)
    #    opt_op_imageNet=opt_imageNet.apply_gradients(gradients_imageNet)
    #

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    opt_Face = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=False)
    opt_imageNet = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9, use_nesterov=False)

    gradients_Face = opt_Face.compute_gradients(loss_Face, gate_gradients=2)
    gradients_imageNet = opt_imageNet.compute_gradients(loss_imageNet, gate_gradients=2)

    gradients = []

    for i in range(len(gradients_Face)):
        if gradients_Face[i][0] is None:
            gradients.append((gradients_imageNet[i][0] * train_ratio_imageNet2Face / (train_ratio_imageNet2Face + 1),
                              gradients_imageNet[i][1]))
        elif gradients_imageNet[i][0] is None:
            gradients.append((gradients_Face[i][0] * 1. / (train_ratio_imageNet2Face + 1), gradients_Face[i][1]))
        else:
            gradients.append(((gradients_Face[i][0] / (train_ratio_imageNet2Face + 1) + gradients_imageNet[i][
                0]) * train_ratio_imageNet2Face / (train_ratio_imageNet2Face + 1), gradients_Face[i][1]))

    opt_op = opt.apply_gradients(gradients)

    #    opt_op_Face=opt.minimize(loss_Face)
    #    opt_op_imageNet=opt.minimize(loss_imageNet)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # %

    for layer_name_tmp in list(W_VGG_Face.keys()):
        sess.run(tf.assign(W_VGG_Face[layer_name_tmp], Weights_VGG_Face[layer_name_tmp]))
        sess.run(tf.assign(b_VGG_Face[layer_name_tmp], bias_VGG_Face[layer_name_tmp]))
        sess.run(tf.assign(W_imageNet[layer_name_tmp], Weights_imageNet[layer_name_tmp]))
        sess.run(tf.assign(b_imageNet[layer_name_tmp], bias_imageNet[layer_name_tmp]))
    for layer_name_tmp in list(W_merged.keys()):
        sess.run(tf.assign(W_merged[layer_name_tmp], Weights_merged[layer_name_tmp]))
        sess.run(tf.assign(b_merged[layer_name_tmp], bias_merged[layer_name_tmp]))
    for layer_name_tmp in list(W_inter_Face.keys()):
        sess.run(tf.assign(W_inter_Face[layer_name_tmp], Weights_inter_Face[layer_name_tmp]))
        sess.run(tf.assign(W_inter_imageNet[layer_name_tmp], Weights_inter_imageNet[layer_name_tmp]))

    # %%
    acc_Face_curr = evaluate(iterator_Face, dataset_Face_val, accuracy_sum_Face, nb_exp_Face_val, EVAL_BATCHES)
    acc_imageNet_curr = evaluate(iterator_imageNet, dataset_imageNet_val, accuracy_sum_imageNet, nb_exp_imageNet_val,
                                 EVAL_BATCHES)

    print('Layer ' + layer_name + ': {:d} kernel/neuron merged, Acc_Face={:%}, Acc_imageNet={:%}'.format(
        merged_nb_dict[layer_name], acc_Face_curr, acc_imageNet_curr))
    # %%
    acc_Face.append(acc_Face_curr)
    acc_imageNet.append(acc_imageNet_curr)
    # %%
    #    _=[]
    #    config_tuple=open(save_dir+layer_name+'_{:d}_merged'.format(merged_nb_dict[layer_name]),'wb')
    #    pickle.dump((Weights_merged, Weights_VGG_Face, Weights_imageNet, Weights_inter_Face, Weights_inter_imageNet,bias_merged, bias_VGG_Face, bias_imageNet, merged_nb_dict,err_mtx,_,_), config_tuple)
    #    config_tuple.close()

    # %%
    if layer_index < 13:
        fine_tune(training_iter_dict[layer_name], 0.5)
    else:
        fine_tune(training_iter_dict[layer_name], 0.5)

    # %%
    acc_Face_curr = evaluate(iterator_Face, dataset_Face_val, accuracy_sum_Face, nb_exp_Face_val, EVAL_BATCHES)
    acc_imageNet_curr = evaluate(iterator_imageNet, dataset_imageNet_val, accuracy_sum_imageNet, nb_exp_imageNet_val,
                                 EVAL_BATCHES)

    print(
        'Layer ' + layer_name + ': {:d} kernel/neuron merged, after fine-tuning, Acc_Face={:%}, Acc_imageNet={:%}'.format(
            merged_nb_dict[layer_name], acc_Face_curr, acc_imageNet_curr))

    # %%
    acc_Face.append(acc_Face_curr)
    acc_imageNet.append(acc_imageNet_curr)

    # %%
    _ = []
    Weights_merged = sess.run(W_merged)
    Weights_VGG_Face = sess.run(W_VGG_Face)
    Weights_imageNet = sess.run(W_imageNet)
    Weights_inter_Face = sess.run(W_inter_Face)
    Weights_inter_imageNet = sess.run(W_inter_imageNet)
    bias_merged = sess.run(b_merged)
    bias_VGG_Face = sess.run(b_VGG_Face)
    bias_imageNet = sess.run(b_imageNet)

    config_tuple = open(save_dir + layer_name + '_{:d}_merged_finetuned'.format(merged_nb_dict[layer_name]), 'wb')
    pickle.dump((Weights_merged, Weights_VGG_Face, Weights_imageNet, Weights_inter_Face, Weights_inter_imageNet,
                 bias_merged, bias_VGG_Face, bias_imageNet, merged_nb_dict, err_mtx, merge_loss, merge_list_all),
                config_tuple)
    config_tuple.close()
    # %
    config_tuple = open(save_dir + layer_name + '_{:d}_merged_acc_recorde'.format(merged_nb_dict[layer_name]), 'wb')
    pickle.dump((acc_Face, acc_imageNet), config_tuple)
    config_tuple.close()

# %%
# layer_index=13
#
# layer_name=layer_name_list[layer_index]
# print('generating layer input'+layer_name)
#
##(Weights_merged, Weights_VGG_Face, Weights_imageNet, Weights_inter_Face, Weights_inter_imageNet,bias_merged, bias_VGG_Face, bias_imageNet, merged_nb_dict,_,_,_)=pickle.load(open(save_dir+layer_name_list[layer_index-1]+'_{:d}_merged_finetuned'.format(merged_nb_dict[layer_name_list[layer_index-1]]),'rb'))
#
# last_kernel_nb=merged_nb_dict[layer_name_list[layer_index-1]]
#
# sess.run(iterator_Face.make_initializer(dataset_Face_train))
# sess.run(iterator_imageNet.make_initializer(dataset_imageNet_train))
#
# layer_input_A=np.zeros((256,last_kernel_nb*7*7))
# layer_input_B=np.zeros((256,last_kernel_nb*7*7))
# for i in range(256//BATCH_SIZE):
#    layer_input_A[i*BATCH_SIZE:(i+1)*BATCH_SIZE]=sess.run(layer_inputs_Face[layer_name], feed_dict={keep_ratio:1})
#    layer_input_B[i*BATCH_SIZE:(i+1)*BATCH_SIZE]=sess.run(layer_inputs_imageNet[layer_name], feed_dict={keep_ratio:1})
#
# np.save(save_dir+'layer_input_A',layer_input_A)
# np.save(save_dir+'layer_input_B',layer_input_B)
