#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 01:35:32 2018

@author: hex
"""
import tensorflow as tf
import copy
import numpy as np

#%%for test
#tf.reset_default_graph()
#merged_nb_dict={
#            'conv1_1':0,
#            'conv1_2':12,
#            'conv2_1':42,
#            'conv2_2':128,
#            'conv3_1':0,
#            'conv3_2':0,
#            'conv3_3':0,
#            'conv4_1':0,
#            'conv4_2':0,
#            'conv4_3':0,
#            'conv5_1':0,
#            'conv5_2':0,
#            'conv5_3':0,
#            'fc6':242,
#            'fc7':4096,
#            }

#%%

def VGG_merged(x_Face, x_imageNet, merged_nb_dict, keep_ratio):
    #%% Create Weights
    layer_name_list=['conv1_1',
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

    before_pool_layer_name_list=['conv1_2',
                                 'conv2_2',
                                 'conv3_3',
                                 'conv4_3']

    normal_conv_layer_name_list=['conv1_1',
                                 'conv2_1',
                                 'conv3_1',
                                 'conv3_2',
                                 'conv4_1',
                                 'conv4_2',
                                 'conv5_1',
                                 'conv5_2',]

    fc_layer_name_list=['fc6','fc7']

    original_shape_dict={
            'conv1_1':[3,3,3,64],
            'conv1_2':[3,3,64,64],
            'conv2_1':[3,3,64,128],
            'conv2_2':[3,3,128,128],
            'conv3_1':[3,3,128,256],
            'conv3_2':[3,3,256,256],
            'conv3_3':[3,3,256,256],
            'conv4_1':[3,3,256,512],
            'conv4_2':[3,3,512,512],
            'conv4_3':[3,3,512,512],
            'conv5_1':[3,3,512,512],
            'conv5_2':[3,3,512,512],
            'conv5_3':[3,3,512,512],
            'fc6':[25088,4096],
            'fc7':[4096,4096],
            'fc8_Face':[4096,40],
            'fc8_imageNet':[4096,1000],
            }
    #%%
    W_merged=dict()
    b_merged=dict()
    W_inter_Face=dict()
#    b_inter_Face=dict()
    W_inter_imageNet=dict()
#    b_inter_imageNet=dict()
    W_VGG_Face=dict()
    b_VGG_Face=dict()
    W_imageNet=dict()
    b_imageNet=dict()

    y_Face=x_Face
    y_imageNet=x_imageNet

    layer_inputs_Face=dict()
    layer_inputs_imageNet=dict()

    layer_inputs_Face['conv1_1']=x_Face
    layer_inputs_imageNet['conv1_1']=x_imageNet

    for i in range(len(layer_name_list)):
        layer_name=layer_name_list[i]

        if i==0: # first layer
            if merged_nb_dict[layer_name]==original_shape_dict[layer_name][3]: # fully merged
                W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=original_shape_dict[layer_name])
                b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=original_shape_dict[layer_name][3])

                y_Face = tf.nn.relu(tf.nn.conv2d(y_Face, W_merged[layer_name] , strides = [1,1,1,1], padding = 'SAME')+ b_merged[layer_name])
                y_imageNet = tf.nn.relu(tf.nn.conv2d(y_imageNet, W_merged[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_merged[layer_name])

                layer_inputs_Face[layer_name_list[i+1]]=y_Face
                layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

            elif merged_nb_dict[layer_name]==0: # not merged
                W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=original_shape_dict[layer_name])
                b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=original_shape_dict[layer_name][3])

                W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=original_shape_dict[layer_name])
                b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=original_shape_dict[layer_name][3])

                y_Face = tf.nn.relu(tf.nn.conv2d(y_Face, W_VGG_Face[layer_name] , strides = [1,1,1,1], padding = 'SAME')+ b_VGG_Face[layer_name])
                y_imageNet = tf.nn.relu(tf.nn.conv2d(y_imageNet, W_imageNet[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet[layer_name])

                layer_inputs_Face[layer_name_list[i+1]]=y_Face
                layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

            else : #partially merged
                shape=copy.deepcopy(original_shape_dict[layer_name])
                shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                shape[3]=shape[3]-merged_nb_dict[layer_name]
                shape_merged[3]=merged_nb_dict[layer_name]

                W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[3])

                W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=shape)
                b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=shape[3])

                W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=shape)
                b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=shape[3])

                layer_inputs_Face[layer_name_list[i+1]]=tf.nn.relu(tf.nn.conv2d(y_Face, W_merged[layer_name] , strides = [1,1,1,1], padding = 'SAME')+ b_merged[layer_name])
                layer_inputs_imageNet[layer_name_list[i+1]]=tf.nn.relu(tf.nn.conv2d(y_imageNet, W_merged[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_merged[layer_name])

                y_Face = tf.concat([layer_inputs_Face[layer_name_list[i+1]],
                                    tf.nn.relu(tf.nn.conv2d(y_Face, W_VGG_Face[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face[layer_name])], axis=3)
                y_imageNet = tf.concat([layer_inputs_imageNet[layer_name_list[i+1]],
                                        tf.nn.relu(tf.nn.conv2d(y_imageNet, W_imageNet[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet[layer_name])], axis=3)



        elif i<13: #all conv layers
            if merged_nb_dict[layer_name]==0: #current layer not merged
                W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=original_shape_dict[layer_name])
                b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=original_shape_dict[layer_name][3])

                W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=original_shape_dict[layer_name])
                b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=original_shape_dict[layer_name][3])

                y_Face = tf.nn.relu(tf.nn.conv2d(y_Face, W_VGG_Face[layer_name] , strides = [1,1,1,1], padding = 'SAME')+ b_VGG_Face[layer_name])
                y_imageNet = tf.nn.relu(tf.nn.conv2d(y_imageNet, W_imageNet[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet[layer_name])

                layer_inputs_Face[layer_name_list[i+1]]=y_Face
                layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

            elif merged_nb_dict[layer_name]==original_shape_dict[layer_name][3]: # current layer fully merged
                if merged_nb_dict[layer_name_list[i-1]]==original_shape_dict[layer_name_list[i-1]][3]: # last layer fully merged, current layer fully merged
                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=original_shape_dict[layer_name])
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=original_shape_dict[layer_name][3])

                    y_Face = tf.nn.relu(tf.nn.conv2d(y_Face, W_merged[layer_name] , strides = [1,1,1,1], padding = 'SAME')+ b_merged[layer_name])
                    y_imageNet = tf.nn.relu(tf.nn.conv2d(y_imageNet, W_merged[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_merged[layer_name])

                    layer_inputs_Face[layer_name_list[i+1]]=y_Face
                    layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

                elif merged_nb_dict[layer_name_list[i-1]]>0: # last layer partially merged, current layer fully merged
                    shape=copy.deepcopy(original_shape_dict[layer_name])
                    shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                    shape_inter=copy.deepcopy(original_shape_dict[layer_name])
                    shape[3]=shape[3]-merged_nb_dict[layer_name]
                    shape_merged[2]=merged_nb_dict[layer_name_list[i-1]]
                    shape_merged[3]=merged_nb_dict[layer_name]
                    shape_inter[2]=shape_inter[2]-merged_nb_dict[layer_name_list[i-1]]
                    shape_inter[3]=merged_nb_dict[layer_name]

                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[3])

                    W_inter_Face[layer_name]=tf.get_variable('W_inter_Face'+layer_name, shape=shape_inter)
#                    b_inter_Face[layer_name]=tf.get_variable('b_inter_Face'+layer_name, shape=shape_inter[3])
                    W_inter_imageNet[layer_name]=tf.get_variable('W_inter_imageNet'+layer_name, shape=shape_inter)
#                    b_inter_imageNet[layer_name]=tf.get_variable('b_inter_imageNet'+layer_name, shape=shape_inter[3])

                    W_mi_Face=tf.concat([W_merged[layer_name],W_inter_Face[layer_name]],axis=2)
                    b_mi_Face=b_merged[layer_name]
                    W_mi_imageNet=tf.concat([W_merged[layer_name],W_inter_imageNet[layer_name]],axis=2)
                    b_mi_imageNet=b_merged[layer_name]

                    y_Face = tf.nn.relu(tf.nn.conv2d(y_Face, W_mi_Face, strides = [1,1,1,1], padding = 'SAME') + b_mi_Face)
                    y_imageNet = tf.nn.relu(tf.nn.conv2d(y_imageNet, W_mi_imageNet, strides = [1,1,1,1], padding = 'SAME') + b_mi_imageNet)

                    layer_inputs_Face[layer_name_list[i+1]]=y_Face
                    layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

                else :#last layer not merged, error!
                    raise NameError('last layer '+layer_name_list[i-1]+' not merged yet, cannot merge current layer')

            else:# current layer partially merged
                if merged_nb_dict[layer_name_list[i-1]]==original_shape_dict[layer_name_list[i-1]][3]: # last layer fully merged, current layer partially merged
                    shape=copy.deepcopy(original_shape_dict[layer_name])
                    shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                    shape[3]=shape[3]-merged_nb_dict[layer_name]
                    shape_merged[3]=merged_nb_dict[layer_name]

                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[3])

                    W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=shape)
                    b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=shape[3])

                    W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=shape)
                    b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=shape[3])

                    layer_inputs_Face[layer_name_list[i+1]]=tf.nn.relu(tf.nn.conv2d(y_Face, W_merged[layer_name] , strides = [1,1,1,1], padding = 'SAME')+ b_merged[layer_name])
                    layer_inputs_imageNet[layer_name_list[i+1]]=tf.nn.relu(tf.nn.conv2d(y_imageNet, W_merged[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_merged[layer_name])

                    y_Face = tf.concat([layer_inputs_Face[layer_name_list[i+1]],
                                        tf.nn.relu(tf.nn.conv2d(y_Face, W_VGG_Face[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face[layer_name])], axis=3)
                    y_imageNet = tf.concat([layer_inputs_imageNet[layer_name_list[i+1]],
                                        tf.nn.relu(tf.nn.conv2d(y_imageNet, W_imageNet[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet[layer_name])], axis=3)

                elif merged_nb_dict[layer_name_list[i-1]]>0: # last layer partially merged, current layer partially merged
                    shape=copy.deepcopy(original_shape_dict[layer_name])
                    shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                    shape_inter=copy.deepcopy(original_shape_dict[layer_name])
                    shape[3]=shape[3]-merged_nb_dict[layer_name]
                    shape_merged[2]=merged_nb_dict[layer_name_list[i-1]]
                    shape_merged[3]=merged_nb_dict[layer_name]
                    shape_inter[2]=shape_inter[2]-merged_nb_dict[layer_name_list[i-1]]
                    shape_inter[3]=merged_nb_dict[layer_name]

                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[3])

                    W_inter_Face[layer_name]=tf.get_variable('W_inter_Face'+layer_name, shape=shape_inter)
#                    b_inter_Face[layer_name]=tf.get_variable('b_inter_Face'+layer_name, shape=shape_inter[3])
                    W_inter_imageNet[layer_name]=tf.get_variable('W_inter_imageNet'+layer_name, shape=shape_inter)
#                    b_inter_imageNet[layer_name]=tf.get_variable('b_inter_imageNet'+layer_name, shape=shape_inter[3])

                    W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=shape)
                    b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=shape[3])
                    W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=shape)
                    b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=shape[3])

                    W_mi_Face=tf.concat([W_merged[layer_name],W_inter_Face[layer_name]],axis=2)
                    b_mi_Face=b_merged[layer_name]
                    W_mi_imageNet=tf.concat([W_merged[layer_name],W_inter_imageNet[layer_name]],axis=2)
                    b_mi_imageNet=b_merged[layer_name]

                    layer_inputs_Face[layer_name_list[i+1]]=tf.nn.relu(tf.nn.conv2d(y_Face, W_mi_Face , strides = [1,1,1,1], padding = 'SAME')+ b_mi_Face)
                    layer_inputs_imageNet[layer_name_list[i+1]]=tf.nn.relu(tf.nn.conv2d(y_imageNet, W_mi_imageNet , strides = [1,1,1,1], padding = 'SAME') + b_mi_imageNet)

                    y_Face = tf.concat([layer_inputs_Face[layer_name_list[i+1]],
                                        tf.nn.relu(tf.nn.conv2d(y_Face, W_VGG_Face[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_VGG_Face[layer_name])], axis=3)
                    y_imageNet = tf.concat([layer_inputs_imageNet[layer_name_list[i+1]],
                                        tf.nn.relu(tf.nn.conv2d(y_imageNet, W_imageNet[layer_name] , strides = [1,1,1,1], padding = 'SAME') + b_imageNet[layer_name])], axis=3)
                else :#last layer not merged, error!
                    raise NameError('last layer '+layer_name_list[i-1]+' not merged yet, cannot merge current layer')

            if layer_name in before_pool_layer_name_list: #max pooling
                layer_inputs_Face[layer_name_list[i+1]]=tf.nn.max_pool(layer_inputs_Face[layer_name_list[i+1]], ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')
                layer_inputs_imageNet[layer_name_list[i+1]]=tf.nn.max_pool(layer_inputs_imageNet[layer_name_list[i+1]], ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

                y_Face = tf.nn.max_pool(y_Face, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')
                y_imageNet = tf.nn.max_pool(y_imageNet, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')

            elif layer_name == 'conv5_3':
                layer_inputs_Face[layer_name_list[i+1]]=tf.contrib.layers.flatten(tf.nn.max_pool(layer_inputs_Face[layer_name_list[i+1]], ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID'))
                layer_inputs_imageNet[layer_name_list[i+1]]=tf.contrib.layers.flatten(tf.nn.max_pool(layer_inputs_imageNet[layer_name_list[i+1]], ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID'))

                y_Face = tf.contrib.layers.flatten(tf.nn.max_pool(y_Face, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID'))
                y_imageNet = tf.contrib.layers.flatten(tf.nn.max_pool(y_imageNet, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID'))

        elif layer_name == 'fc6':
            if merged_nb_dict[layer_name]==0: # current layer not merged
                W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=original_shape_dict[layer_name])
                b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=original_shape_dict[layer_name][1])

                W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=original_shape_dict[layer_name])
                b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=original_shape_dict[layer_name][1])

                layer_inputs_Face[layer_name_list[i+1]]=y_Face = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_Face, W_VGG_Face[layer_name]), b_VGG_Face[layer_name])),keep_prob=keep_ratio)
                layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_imageNet[layer_name]), b_imageNet[layer_name])),keep_prob=keep_ratio)

            elif merged_nb_dict[layer_name]==original_shape_dict[layer_name][1]: # current layer fully merged
                if merged_nb_dict[layer_name_list[i-1]]==original_shape_dict[layer_name_list[i-1]][3]: # last layer fully merged, current layer fully merged
                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=original_shape_dict[layer_name])
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=original_shape_dict[layer_name][1])

                    y_Face = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_Face, W_merged[layer_name]), b_merged[layer_name])),keep_prob=keep_ratio)
                    y_imageNet = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_merged[layer_name]), b_merged[layer_name])),keep_prob=keep_ratio)

                    layer_inputs_Face[layer_name_list[i+1]]=y_Face
                    layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

                elif merged_nb_dict[layer_name_list[i-1]]>0: # last layer partially merged, current layer fully merged
                    shape=copy.deepcopy(original_shape_dict[layer_name])
                    shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                    shape_inter=copy.deepcopy(original_shape_dict[layer_name])

                    shape[1]=shape[1]-merged_nb_dict[layer_name]
                    shape_merged[0]=7*7*merged_nb_dict[layer_name_list[i-1]]
                    shape_merged[1]=merged_nb_dict[layer_name]
                    shape_inter[0]=shape_inter[0]-shape_merged[0]
                    shape_inter[1]=merged_nb_dict[layer_name]

                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[1])

                    W_inter_Face[layer_name]=tf.get_variable('W_inter_Face'+layer_name, shape=shape_inter)
#                    b_inter_Face[layer_name]=tf.get_variable('b_inter_Face'+layer_name, shape=shape_inter[1])
                    W_inter_imageNet[layer_name]=tf.get_variable('W_inter_imageNet'+layer_name, shape=shape_inter)
#                    b_inter_imageNet[layer_name]=tf.get_variable('b_inter_imageNet'+layer_name, shape=shape_inter[1])

                    W_mi_Face=tf.concat([W_merged[layer_name],W_inter_Face[layer_name]],axis=0)
                    b_mi_Face=b_merged[layer_name]
                    W_mi_imageNet=tf.concat([W_merged[layer_name],W_inter_imageNet[layer_name]],axis=0)
                    b_mi_imageNet=b_merged[layer_name]

                    y_Face = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_Face, W_mi_Face), b_mi_Face)),keep_prob=keep_ratio)
                    y_imageNet = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_mi_imageNet), b_mi_imageNet)),keep_prob=keep_ratio)

                    layer_inputs_Face[layer_name_list[i+1]]=y_Face
                    layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

                else :#last layer not merged, error!
                    raise NameError('last layer '+layer_name_list[i-1]+' not merged yet, cannot merge current layer')

            else:#partially merged
                if merged_nb_dict[layer_name_list[i-1]]==original_shape_dict[layer_name_list[i-1]][3]: # last layer fully merged, current layer partially merged
                    shape=copy.deepcopy(original_shape_dict[layer_name])
                    shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                    shape[1]=shape[1]-merged_nb_dict[layer_name]
                    shape_merged[1]=merged_nb_dict[layer_name]

                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[1])

                    W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=shape)
                    b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=shape[1])

                    W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=shape)
                    b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=shape[1])

                    layer_inputs_Face[layer_name_list[i+1]]=tf.nn.relu(tf.add(tf.matmul(y_Face, W_merged[layer_name]), b_merged[layer_name]))
                    layer_inputs_imageNet[layer_name_list[i+1]]=tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_merged[layer_name]), b_merged[layer_name]))

                    y_Face = tf.nn.dropout(tf.concat([layer_inputs_Face[layer_name_list[i+1]],
                                        tf.nn.relu(tf.add(tf.matmul(y_Face, W_VGG_Face[layer_name]), b_VGG_Face[layer_name]))], axis=1),keep_prob=keep_ratio)
                    y_imageNet = tf.nn.dropout(tf.concat([layer_inputs_imageNet[layer_name_list[i+1]],
                                            tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_imageNet[layer_name]), b_imageNet[layer_name]))], axis=1),keep_prob=keep_ratio)

                elif merged_nb_dict[layer_name_list[i-1]]>0: # last layer partially merged, current layer partially merged
                    shape=copy.deepcopy(original_shape_dict[layer_name])
                    shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                    shape_inter=copy.deepcopy(original_shape_dict[layer_name])

                    shape[1]=shape[1]-merged_nb_dict[layer_name]
                    shape_merged[0]=7*7*merged_nb_dict[layer_name_list[i-1]]
                    shape_merged[1]=merged_nb_dict[layer_name]
                    shape_inter[0]=shape_inter[0]-shape_merged[0]
                    shape_inter[1]=merged_nb_dict[layer_name]

                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[1])

                    W_inter_Face[layer_name]=tf.get_variable('W_inter_Face'+layer_name, shape=shape_inter)
#                    b_inter_Face[layer_name]=tf.get_variable('b_inter_Face'+layer_name, shape=shape_inter[1])
                    W_inter_imageNet[layer_name]=tf.get_variable('W_inter_imageNet'+layer_name, shape=shape_inter)
#                    b_inter_imageNet[layer_name]=tf.get_variable('b_inter_imageNet'+layer_name, shape=shape_inter[1])

                    W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=shape)
                    b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=shape[1])
                    W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=shape)
                    b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=shape[1])

                    W_mi_Face=tf.concat([W_merged[layer_name],W_inter_Face[layer_name]],axis=0)
                    b_mi_Face=b_merged[layer_name]
                    W_mi_imageNet=tf.concat([W_merged[layer_name],W_inter_imageNet[layer_name]],axis=0)
                    b_mi_imageNet=b_merged[layer_name]

                    layer_inputs_Face[layer_name_list[i+1]]=tf.nn.relu(tf.add(tf.matmul(y_Face, W_mi_Face), b_mi_Face))
                    layer_inputs_imageNet[layer_name_list[i+1]]=tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_mi_imageNet), b_mi_imageNet))

                    y_Face = tf.nn.dropout(tf.concat([layer_inputs_Face[layer_name_list[i+1]],
                                        tf.nn.relu(tf.add(tf.matmul(y_Face, W_VGG_Face[layer_name]), b_VGG_Face[layer_name]))], axis=1),keep_prob=keep_ratio)
                    y_imageNet = tf.nn.dropout(tf.concat([layer_inputs_imageNet[layer_name_list[i+1]],
                                            tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_imageNet[layer_name]), b_imageNet[layer_name]))], axis=1),keep_prob=keep_ratio)
                else :#last layer not merged, error!
                    raise NameError('last layer '+layer_name_list[i-1]+' not merged yet, cannot merge current layer')

        elif i<15:
            if merged_nb_dict[layer_name]==0: # current layer not merged
                W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=original_shape_dict[layer_name])
                b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=original_shape_dict[layer_name][1])

                W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=original_shape_dict[layer_name])
                b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=original_shape_dict[layer_name][1])

                y_Face = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_Face, W_VGG_Face[layer_name]), b_VGG_Face[layer_name])),keep_prob=keep_ratio)
                y_imageNet = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_imageNet[layer_name]), b_imageNet[layer_name])),keep_prob=keep_ratio)

                layer_inputs_Face[layer_name_list[i+1]]=y_Face
                layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

            elif merged_nb_dict[layer_name]==original_shape_dict[layer_name][1]: # current layer fully merged
                if merged_nb_dict[layer_name_list[i-1]]==original_shape_dict[layer_name_list[i-1]][1]: # last layer fully merged, current layer fully merged
                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=original_shape_dict[layer_name])
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=original_shape_dict[layer_name][1])

                    y_Face = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_Face, W_merged[layer_name]), b_merged[layer_name])),keep_prob=keep_ratio)
                    y_imageNet = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_merged[layer_name]), b_merged[layer_name])),keep_prob=keep_ratio)

                    layer_inputs_Face[layer_name_list[i+1]]=y_Face
                    layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

                elif merged_nb_dict[layer_name_list[i-1]]>0: # last layer partially merged, current layer fully merged
                    shape=copy.deepcopy(original_shape_dict[layer_name])
                    shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                    shape_inter=copy.deepcopy(original_shape_dict[layer_name])

                    shape[1]=shape[1]-merged_nb_dict[layer_name]
                    shape_merged[0]=merged_nb_dict[layer_name_list[i-1]]
                    shape_merged[1]=merged_nb_dict[layer_name]
                    shape_inter[0]=shape_inter[0]-shape_merged[0]
                    shape_inter[1]=merged_nb_dict[layer_name]

                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[1])

                    W_inter_Face[layer_name]=tf.get_variable('W_inter_Face'+layer_name, shape=shape_inter)
#                    b_inter_Face[layer_name]=tf.get_variable('b_inter_Face'+layer_name, shape=shape_inter[1])
                    W_inter_imageNet[layer_name]=tf.get_variable('W_inter_imageNet'+layer_name, shape=shape_inter)
#                    b_inter_imageNet[layer_name]=tf.get_variable('b_inter_imageNet'+layer_name, shape=shape_inter[1])

                    W_mi_Face=tf.concat([W_merged[layer_name],W_inter_Face[layer_name]],axis=0)
                    b_mi_Face=b_merged[layer_name]
                    W_mi_imageNet=tf.concat([W_merged[layer_name],W_inter_imageNet[layer_name]],axis=0)
                    b_mi_imageNet=b_merged[layer_name]

                    y_Face = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_Face, W_mi_Face), b_mi_Face)),keep_prob=keep_ratio)
                    y_imageNet = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_mi_imageNet), b_mi_imageNet)),keep_prob=keep_ratio)

                    layer_inputs_Face[layer_name_list[i+1]]=y_Face
                    layer_inputs_imageNet[layer_name_list[i+1]]=y_imageNet

                else :#last layer not merged, error!
                    raise NameError('last layer '+layer_name_list[i-1]+' not merged yet, cannot merge current layer')

            else:#partially merged
                if merged_nb_dict[layer_name_list[i-1]]==original_shape_dict[layer_name_list[i-1]][1]: # last layer fully merged, current layer partially merged
                    shape=copy.deepcopy(original_shape_dict[layer_name])
                    shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                    shape[1]=shape[1]-merged_nb_dict[layer_name]
                    shape_merged[1]=merged_nb_dict[layer_name]

                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[1])

                    W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=shape)
                    b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=shape[1])

                    W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=shape)
                    b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=shape[1])

                    layer_inputs_Face[layer_name_list[i+1]]=tf.nn.relu(tf.add(tf.matmul(y_Face, W_merged[layer_name]), b_merged[layer_name]))
                    layer_inputs_imageNet[layer_name_list[i+1]]=tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_merged[layer_name]), b_merged[layer_name]))

                    y_Face = tf.nn.dropout(tf.concat([layer_inputs_Face[layer_name_list[i+1]],
                                        tf.nn.relu(tf.add(tf.matmul(y_Face, W_VGG_Face[layer_name]), b_VGG_Face[layer_name]))], axis=1),keep_prob=keep_ratio)
                    y_imageNet = tf.nn.dropout(tf.concat([layer_inputs_imageNet[layer_name_list[i+1]],
                                            tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_imageNet[layer_name]), b_imageNet[layer_name]))], axis=1),keep_prob=keep_ratio)

                elif merged_nb_dict[layer_name_list[i-1]]>0: # last layer partially merged, current layer partially merged
                    shape=copy.deepcopy(original_shape_dict[layer_name])
                    shape_merged=copy.deepcopy(original_shape_dict[layer_name])
                    shape_inter=copy.deepcopy(original_shape_dict[layer_name])

                    shape[1]=shape[1]-merged_nb_dict[layer_name]
                    shape_merged[0]=merged_nb_dict[layer_name_list[i-1]]
                    shape_merged[1]=merged_nb_dict[layer_name]
                    shape_inter[0]=shape_inter[0]-shape_merged[0]
                    shape_inter[1]=merged_nb_dict[layer_name]

                    W_merged[layer_name]=tf.get_variable('W_merged_'+layer_name, shape=shape_merged)
                    b_merged[layer_name]=tf.get_variable('b_merged_'+layer_name, shape=shape_merged[1])

                    W_inter_Face[layer_name]=tf.get_variable('W_inter_Face'+layer_name, shape=shape_inter)
#                    b_inter_Face[layer_name]=tf.get_variable('b_inter_Face'+layer_name, shape=shape_inter[1])
                    W_inter_imageNet[layer_name]=tf.get_variable('W_inter_imageNet'+layer_name, shape=shape_inter)
#                    b_inter_imageNet[layer_name]=tf.get_variable('b_inter_imageNet'+layer_name, shape=shape_inter[1])

                    W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_'+layer_name, shape=shape)
                    b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_'+layer_name, shape=shape[1])
                    W_imageNet[layer_name]=tf.get_variable('W_imageNet_'+layer_name, shape=shape)
                    b_imageNet[layer_name]=tf.get_variable('b_imageNet_'+layer_name, shape=shape[1])

                    W_mi_Face=tf.concat([W_merged[layer_name],W_inter_Face[layer_name]],axis=0)
                    b_mi_Face=b_merged[layer_name]
                    W_mi_imageNet=tf.concat([W_merged[layer_name],W_inter_imageNet[layer_name]],axis=0)
                    b_mi_imageNet=b_merged[layer_name]

                    layer_inputs_Face[layer_name_list[i+1]]=tf.nn.relu(tf.add(tf.matmul(y_Face, W_mi_Face), b_mi_Face))
                    layer_inputs_imageNet[layer_name_list[i+1]]=tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_mi_imageNet), b_mi_imageNet))

                    y_Face = tf.nn.dropout(tf.concat([layer_inputs_Face[layer_name_list[i+1]],
                                        tf.nn.relu(tf.add(tf.matmul(y_Face, W_VGG_Face[layer_name]), b_VGG_Face[layer_name]))], axis=1),keep_prob=keep_ratio)
                    y_imageNet = tf.nn.dropout(tf.concat([layer_inputs_imageNet[layer_name_list[i+1]],
                                            tf.nn.relu(tf.add(tf.matmul(y_imageNet, W_imageNet[layer_name]), b_imageNet[layer_name]))], axis=1),keep_prob=keep_ratio)
                else :#last layer not merged, error!
                    raise NameError('last layer '+layer_name_list[i-1]+' not merged yet, cannot merge current layer')

        elif layer_name=='fc8':
            W_VGG_Face[layer_name]=tf.get_variable('W_VGG_Face_fc8', shape=original_shape_dict['fc8_Face'])
            b_VGG_Face[layer_name]=tf.get_variable('b_VGG_Face_fc8', shape=original_shape_dict['fc8_Face'][1])

            W_imageNet[layer_name]=tf.get_variable('W_imageNet_fc8', shape=original_shape_dict['fc8_imageNet'])
            b_imageNet[layer_name]=tf.get_variable('b_imageNet_fc8', shape=original_shape_dict['fc8_imageNet'][1])

            y_Face = tf.add(tf.matmul(y_Face, W_VGG_Face[layer_name]), b_VGG_Face[layer_name])
            y_imageNet = tf.add(tf.matmul(y_imageNet, W_imageNet[layer_name]), b_imageNet[layer_name])
#%%

    return y_Face, y_imageNet, W_merged, b_merged, W_inter_Face, W_VGG_Face, b_VGG_Face, layer_inputs_Face, W_inter_imageNet, W_imageNet, b_imageNet, layer_inputs_imageNet