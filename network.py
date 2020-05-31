# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:09:22 2018

@author: chlian
"""

import numpy as np
import scipy.io as sio
import keras
from keras.models import Model
from keras import backend as K
from keras import layers

from keras.layers import Input, Flatten
from keras.layers.convolutional import Conv3D, UpSampling3D, Conv1D
from keras.layers.pooling import MaxPooling3D, AveragePooling3D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.merge import Concatenate, Average, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Dropout, Reshape, Permute
from keras import regularizers
from keras.regularizers import Regularizer

def flatten_model(model_nested):
    layers_flat = []
    for layer in model_nested.layers:
        try:
            layers_flat.extend(layer.layers)
        except AttributeError:
            layers_flat.append(layer)
    #model_flat = keras.models.Sequential(layers_flat)
    return layers_flat

class L2Normalization(layers.Layer):
    
    def comput_output_shape(self, input_shape):
        return tuple(input_shape)
    
    def call(self, inputs):
        inputs = K.l2_normalize(inputs, axis=1)
        return inputs

def acc(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

class StructuredSparse(Regularizer):
    
    def __init__(self, C=0.001):
        self.C = K.cast_to_floatx(C)
    
    def __call__(self, kernel_matrix):
        #const_coeff = np.sqrt(K.int_shape(kernel_matrix)[-1] * K.int_shape(kernel_matrix)[-2])
        return self.C * \
               K.sum(K.sqrt(K.sum(K.sum(K.square(kernel_matrix), axis=-1), axis=-1))) 

    def get_config(self):
        return {'C': float(self.C)}


class CAM(object):
    '''
    Backbone FCN to produce class activation maps (CAMs)
    '''
    def __init__(self, 
                 image_size,
                 num_chns, 
                 num_outputs, 
                 feature_depth,
                 with_bn=True,
                 with_dropout=True,
                 drop_prob=0.5,
                 trn_flag=True):
        self.num_chns = num_chns
        self.num_outputs = num_outputs
        self.img_size = image_size
        self.feature_depth = feature_depth
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop_prob = drop_prob
        self.trn_flag = trn_flag
            
    def forward(self, optimizer='Adam', metrics=['accuracy']):
        inputs = Input((self.num_chns, self.img_size[0], 
                        self.img_size[1], self.img_size[2]))
                        
        conv1 = Conv3D(filters=self.feature_depth[0], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv1')(inputs)
        if self.with_bn:
            conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)
        
        conv2 = Conv3D(filters=self.feature_depth[1], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv2')(conv1)
        if self.with_bn:
            conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)
        if self.with_dropout:
            conv2 = Dropout(self.drop_prob)(conv2)

        pool1 = Conv3D(filters=self.feature_depth[1], 
                       kernel_size=[2, 2, 2],
                       strides=[2, 2, 2],
                       padding='valid', 
                       data_format='channels_first', 
                       name='pool1')(conv2)
        if self.with_bn:
            pool1 = BatchNormalization(axis=1)(pool1)
        pool1 = Activation('relu')(pool1)
                   
        conv3 = Conv3D(filters=self.feature_depth[2], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv3')(pool1)
        if self.with_bn:
            conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = Activation('relu')(conv3)

        conv4 = Conv3D(filters=self.feature_depth[3], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv4')(conv3)
        if self.with_bn:
            conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = Activation('relu')(conv4)
        if self.with_dropout:
            conv4 = Dropout(self.drop_prob)(conv4)

        pool2 = Conv3D(filters=self.feature_depth[3], 
                       kernel_size=[2, 2, 2],
                       strides=[2, 2, 2],
                       padding='valid', 
                       data_format='channels_first', 
                       name='pool2')(conv4)
        if self.with_bn:
            pool2 = BatchNormalization(axis=1)(pool2)
        pool2 = Activation('relu')(pool2)
            
        conv5 = Conv3D(filters=self.feature_depth[4], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv5')(pool2)
        if self.with_bn:
            conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = Activation('relu')(conv5)

        conv6 = Conv3D(filters=self.feature_depth[5], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first',
                       name='conv6')(conv5)
        if self.with_bn:
            conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = Activation('relu')(conv6)
        if self.with_dropout:
            conv6 = Dropout(self.drop_prob)(conv6)

        pool3 = Conv3D(filters=self.feature_depth[5], 
                       kernel_size=[2, 2, 2],
                       strides=[2, 2, 2],
                       padding='valid', 
                       data_format='channels_first', 
                       name='pool3')(conv6)
        if self.with_bn:
            pool3 = BatchNormalization(axis=1)(pool3)
        pool3 = Activation('relu')(pool3)

        conv7 = Conv3D(filters=self.feature_depth[6], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first',
                       name='conv7')(pool3)
        if self.with_bn:
            conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = Activation('relu')(conv7)

        conv8 = Conv3D(filters=self.feature_depth[7], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first',
                       name='conv8')(conv7)
        if self.with_bn:
            conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = Activation('relu')(conv8)

        transfer = Conv3D(filters=64,
                           kernel_size=[1, 1, 1],
                           padding='same', 
                           data_format='channels_first',
                           name='transfer')(conv8)
        if self.with_bn:
            transfer = BatchNormalization(axis=1)(transfer)
        transfer = Activation('relu')(transfer)
        if self.with_dropout:
            transfer = Dropout(self.drop_prob)(transfer)   
        
        score = AveragePooling3D(pool_size=(self.img_size[0]/8, 
                                            self.img_size[1]/8,
                                            self.img_size[2]/8), 
                              data_format='channels_first', 
                              name='gap')(transfer)
        score = Flatten(name='score')(score)
        
        prob = Dense(units=1, activation='sigmoid', use_bias=False,
                     name='subject_outputs')(score)
        if self.trn_flag:
          model = Model(inputs=inputs, outputs=[prob])
          model.compile(loss='binary_crossentropy', optimizer=optimizer, 
                      metrics=['accuracy'])
        else:
          model = Model(inputs=inputs, outputs=[prob, transfer])
        
        return model  


class HierachicalNet(object):
    '''
    Local-branch H-FCN with patch locations INITIALLY defined by the mean CAM
    '''
    def __init__(self, 
                 patch_size,
                 num_patches,
                 num_neighbors,
                 neighbor_matrix,
                 num_chns, 
                 num_outputs, 
                 feature_depth,
                 num_region_features=64,
                 num_subject_features=64,
                 with_bn=True,
                 with_dropout=True,
                 drop_prob=0.5,
                 region_sparse=0.0,
                 subject_sparse=0.0):
        self.num_chns = num_chns
        self.num_outputs = num_outputs
        self.num_patches = num_patches
        
        self.num_neighbors = num_neighbors 
        self.nn_mat = np.array(neighbor_matrix)[:, :num_neighbors]
        
        self.patch_size = patch_size
        self.feature_depth = feature_depth
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop_prob = drop_prob
            
        self.num_region_features = num_region_features
        self.num_subject_features = num_subject_features
        
        self.region_sparse = region_sparse
        self.subject_sparse = subject_sparse

    def get_global_net(self):
        
        embed_net = self.base_net()
        
        # inputs
        inputs = []
        for i_input in range(self.num_patches):
            input_name = 'input_{0}'.format(i_input+1)
            inputs.append(Input((self.num_chns, self.patch_size, 
                        self.patch_size, self.patch_size), 
                        name=input_name))
        
        #+++++++++++++++++++++++++++++#                      
        ##   patch-level processing   #
        #+++++++++++++++++++++++++++++# 
        patch_features_list, patch_probs_list = [], []
        for i_input in range(self.num_patches):
            feature_map, class_prob = embed_net(inputs[i_input])
            patch_features_list.append(feature_map)
            patch_probs_list.append(class_prob)
            
        patch_outputs = Concatenate(name='patch_outputs', 
                                    axis=1)(patch_probs_list)
                                    
        #+++++++++++++++++++++++++++++++#                      
        ##   region-level processing   ## 
        #+++++++++++++++++++++++++++++++#
        region_features_list, region_probs_list = [], []
        i_region = 1
        for i_input in range(self.num_patches):
            nn_features, nn_probs = [], []
            for i_neighbor in range(self.num_neighbors):
                nn_features.append(patch_features_list[self.nn_mat[i_input, i_neighbor]])
                nn_probs.append(patch_probs_list[self.nn_mat[i_input, i_neighbor]])   
            region_input_features = Concatenate(axis=1)(nn_features)
            region_input_probs = Concatenate(axis=1)(nn_probs)
            region_input_probs = Reshape([self.num_neighbors, 1])(region_input_probs)
            
            in_name = 'region_input_{0}'.format(i_region)
            region_input = Concatenate(name=in_name, 
                                       axis=-1)([region_input_probs, region_input_features])
            conv_name = 'region_conv_{0}'.format(i_region)
            region_feature = Conv1D(filters=self.num_region_features,
                                    kernel_size=self.num_neighbors,
                                    kernel_regularizer=StructuredSparse(self.region_sparse),
                                    padding='valid', 
                                    name=conv_name)(region_input)
            if self.with_bn:
                region_feature = BatchNormalization(axis=-1)(region_feature)
            region_feature = Activation('relu')(region_feature)
            if self.with_dropout:
                region_feature = Dropout(self.drop_prob)(region_feature)
            ot_name = 'region_prob_{0}'.format(i_region)
            region_prob = Dense(units=1, activation='sigmoid',
                                name=ot_name)(Flatten()(region_feature))
                                
            region_features_list.append(region_feature)
            region_probs_list.append(region_prob)
            
            i_region += 1          
            
        region_outputs = Concatenate(name='region_outputs', 
                                     axis=1)(region_probs_list)
        region_features = Concatenate(name='region_features', 
                                      axis=1)(region_features_list)
        region_probs = Reshape([self.num_patches, 1], 
                               name='region_probs')(region_outputs)
          
        region_feat_prob = Concatenate(name='region_features_probs', 
                                      axis=-1)([region_probs, region_features]) 

        #+++++++++++++++++++++++++++++++#                      
        ##   subject-level processing   #
        #+++++++++++++++++++++++++++++++#
        subject_feature = Conv1D(filters=self.num_subject_features,
                                 kernel_size=self.num_patches,
                                 kernel_regularizer=StructuredSparse(self.subject_sparse),
                                 padding='valid', 
                                 name='subject_conv')(region_feat_prob)
        if self.with_bn:
            subject_feature = BatchNormalization(axis=-1)(subject_feature)
        subject_feature = Activation('relu')(subject_feature)
        if self.with_dropout:
            subject_feature = Dropout(self.drop_prob)(subject_feature)
            
        # subject-level units
        subject_units = Flatten(name='subject_level_units')(subject_feature)
        
        #+++++++++++#                      
        #   Output  #
        #+++++++++++#
        subject_outputs = Dense(units=1, activation='sigmoid',
                                name='subject_outputs')(subject_units)
        
        outputs = [patch_outputs, region_outputs, subject_outputs] 
        model = Model(inputs=inputs, outputs=outputs)
      
        return model
    
    def base_net(self):
        
        """ Input with channel first"""
        inputs = Input((self.num_chns, self.patch_size, 
                        self.patch_size, self.patch_size))
                        
        """ 1st convolution"""                
        conv1 = Conv3D(filters=self.feature_depth[0], 
                       kernel_size=[4, 4, 4],
                       padding='valid', 
                       data_format='channels_first')(inputs)
        if self.with_bn:
            conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)

        """ 2nd convolution"""
        conv2 = Conv3D(filters=self.feature_depth[1], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(conv1)
        if self.with_bn:
            conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)
 
        """ pooling 1"""
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), 
                             data_format='channels_first')(conv2)
 
        """ 3rd convolution"""
        conv3 = Conv3D(filters=self.feature_depth[2], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(pool1)
        if self.with_bn:
            conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = Activation('relu')(conv3)

        """ 4th convolution"""
        conv4 = Conv3D(filters=self.feature_depth[3], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(conv3)
        if self.with_bn:
            conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = Activation('relu')(conv4)
        
        """ pooling 2"""
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), 
                             data_format='channels_first')(conv4)
                             
        """ 5th convolution"""
        conv5 = Conv3D(filters=self.feature_depth[4], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(pool2)
        if self.with_bn:
            conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = Activation('relu')(conv5)

        """ 6th convolution"""
        conv6 = Conv3D(filters=self.feature_depth[5], 
                       kernel_size=[1, 1, 1],
                       padding='valid', 
                       data_format='channels_first')(conv5)
        if self.with_bn:
            conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = Activation('relu')(conv6)
        if self.with_dropout:
            conv6 = Dropout(self.drop_prob)(conv6)
        
        feature_map = Reshape((1, self.feature_depth[-1]),
                              name='patch_features')(conv6)
        class_prob = Dense(units=1, activation='sigmoid',
                           name='patch_prob')(Flatten()(feature_map))
        
        model = Model(inputs=inputs, 
                      outputs=[feature_map, class_prob], 
                      name='base_net')

        model.summary()
        
        return model

class PruHNet(object):
    '''
    Local branch of HybNet
    '''
    def __init__(self, 
                 patch_size,
                 num_patches,
                 pruned_nns,
                 num_chns, 
                 num_outputs, 
                 feature_depth,
                 num_region_features=64,
                 num_subject_features=64,
                 with_bn=True,
                 with_dropout=True,
                 drop_prob=0.5,
                 region_sparse=0.0,
                 subject_sparse=0.0):
        self.num_chns = num_chns
        self.num_outputs = num_outputs
        self.num_patches = num_patches
        
        self.nn_mat = pruned_nns
        self.num_regions = np.shape(pruned_nns)[0]
        
        self.patch_size = patch_size
        self.feature_depth = feature_depth
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop_prob = drop_prob
            
        self.num_region_features = num_region_features
        self.num_subject_features = num_subject_features
        
        self.region_sparse = region_sparse
        self.subject_sparse = subject_sparse

    def get_pruned_net(self):
        
        embed_net = self.base_net()
        
        # inputs
        inputs = []
        for i_input in range(self.num_patches):
            input_name = 'input_{0}'.format(i_input+1)
            inputs.append(Input((self.num_chns, self.patch_size, 
                        self.patch_size, self.patch_size), 
                        name=input_name))
        
        #+++++++++++++++++++++++++++++#                      
        ##   patch-level processing   #
        #+++++++++++++++++++++++++++++# 
        patch_features_list, patch_probs_list = [], []
        for i_input in range(self.num_patches):
            feature_map, class_prob = embed_net(inputs[i_input])
            patch_features_list.append(feature_map)
            patch_probs_list.append(class_prob)     
        patch_outputs = Concatenate(name='patch_outputs', 
                                    axis=1)(patch_probs_list)
                                    
        #+++++++++++++++++++++++++++++++#                      
        ##   region-level processing   ## 
        #+++++++++++++++++++++++++++++++#
        region_features_list, region_probs_list = [], []
        for i_region in range(self.num_regions):
            nn_idxs = self.nn_mat[i_region][0][0].tolist()
            nn_features, nn_probs = [], []
            num_neighbors = len(nn_idxs)
            for i_neighbor in nn_idxs:
                nn_features.append(patch_features_list[i_neighbor])
                nn_probs.append(patch_probs_list[i_neighbor])
            region_input_features = Concatenate(axis=1)(nn_features)
            region_input_probs = Concatenate(axis=1)(nn_probs)
            region_input_probs = Reshape([num_neighbors, 1])(region_input_probs)
            
            in_name = 'region_input_{0}'.format(i_region+1)
            region_input = Concatenate(name=in_name, 
                                       axis=-1)([region_input_probs, region_input_features])
            conv_name = 'region_conv_{0}'.format(i_region+1)
            region_feature = Conv1D(filters=self.num_region_features,
                                    kernel_size=2,
                                    padding='same', 
                                    name=conv_name)(region_input)
            if self.with_bn:
                region_feature = BatchNormalization(axis=-1)(region_feature)
            region_feature = Activation('relu')(region_feature)
            if self.with_dropout:
                region_feature = Dropout(self.drop_prob)(region_feature)
            
            gap_name = 'region_gap_{0}'.format(i_region+1)
            region_feature = AveragePooling1D(pool_size=num_neighbors,
                                            name=gap_name)(region_feature)

            ot_name = 'region_prob_{0}'.format(i_region+1)
            region_prob = Dense(units=1, activation='sigmoid',
                                name=ot_name)(Flatten()(region_feature))
                                
            region_features_list.append(region_feature)
            region_probs_list.append(region_prob)        
            
        region_outputs = Concatenate(name='region_outputs', 
                                     axis=1)(region_probs_list)
        region_features = Concatenate(name='region_features', 
                                      axis=1)(region_features_list)
        region_probs = Reshape([self.num_regions, 1], 
                               name='region_probs')(region_outputs)
          
        region_feat_prob = Concatenate(name='region_features_probs', 
                                      axis=-1)([region_probs, region_features]) 

        #+++++++++++++++++++++++++++++++#                      
        ##   subject-level processing   #
        #+++++++++++++++++++++++++++++++#
        subject_feature = Conv1D(filters=self.num_subject_features,
                                 kernel_size=2,
                                 padding='same', 
                                 name='subject_conv_1')(region_feat_prob)
        if self.with_bn:
            subject_feature = BatchNormalization(axis=-1)(subject_feature)
        subject_feature = Activation('relu')(subject_feature)

        subject_feature = Conv1D(filters=self.num_subject_features / 2,
                                 kernel_size=2,
                                 padding='same', 
                                 name='subject_conv_2')(region_feat_prob)
        if self.with_bn:
            subject_feature = BatchNormalization(axis=-1)(subject_feature)
        subject_feature = Activation('relu')(subject_feature)

        if self.with_dropout:
            subject_feature = Dropout(self.drop_prob)(subject_feature)
        
        subject_feature = AveragePooling1D(pool_size=self.num_regions,
                                            name='subject_gap')(subject_feature)

        # subject-level units
        subject_units = Flatten(name='subject_level_units')(subject_feature)
        
        #+++++++++++#                      
        #   OUTPUT  #
        #+++++++++++#
        subject_outputs = Dense(units=1, activation='sigmoid',
                                name='subject_outputs')(subject_units)
        
        outputs = [patch_outputs, region_outputs, subject_outputs] 
        model = Model(inputs=inputs, outputs=outputs)
      
        return model
    
    def base_net(self):
        
        """ Input with channel first"""
        inputs = Input((self.num_chns, self.patch_size, 
                        self.patch_size, self.patch_size))
                        
        """ 1st convolution"""                
        conv1 = Conv3D(filters=self.feature_depth[0], 
                       kernel_size=[4, 4, 4],
                       padding='valid', 
                       data_format='channels_first')(inputs)
        if self.with_bn:
            conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)

        """ 2nd convolution"""
        conv2 = Conv3D(filters=self.feature_depth[1], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(conv1)
        if self.with_bn:
            conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)
 
        """ pooling 1"""
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), 
                             data_format='channels_first')(conv2)
 
        """ 3rd convolution"""
        conv3 = Conv3D(filters=self.feature_depth[2], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(pool1)
        if self.with_bn:
            conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = Activation('relu')(conv3)

        """ 4th convolution"""
        conv4 = Conv3D(filters=self.feature_depth[3], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(conv3)
        if self.with_bn:
            conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = Activation('relu')(conv4)
        
        """ pooling 2"""
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), 
                             data_format='channels_first')(conv4)
                             
        """ 5th convolution"""
        conv5 = Conv3D(filters=self.feature_depth[4], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(pool2)
        if self.with_bn:
            conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = Activation('relu')(conv5)

        """ 6th convolution"""
        conv6 = Conv3D(filters=self.feature_depth[5], 
                       kernel_size=[1, 1, 1],
                       padding='valid', 
                       data_format='channels_first')(conv5)
        if self.with_bn:
            conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = Activation('relu')(conv6)
        if self.with_dropout:
            conv6 = Dropout(self.drop_prob)(conv6)
     
        feature_map = Reshape((1, self.feature_depth[-1]),
                              name='patch_features')(conv6)
     
        class_prob = Dense(units=1, activation='sigmoid',
                           name='patch_prob')(Flatten()(feature_map))
        
        model = Model(inputs=inputs, 
                      outputs=[feature_map, class_prob], 
                      name='base_net')
        return model


class AttGatedNet(object):
    '''
    Global branch of HybNet
    '''
    def __init__(self, 
                 image_size,
                 num_chns, 
                 num_outputs, 
                 feature_depth,
                 lr_chns=64,
                 gb_chns=64,
                 with_bn=True,
                 with_dropout=True,
                 drop_prob=0.5,
                 trn_flag=True):
        self.num_chns = num_chns
        self.num_outputs = num_outputs
        self.img_size = image_size
        self.feature_depth = feature_depth
        self.lr_chns = lr_chns
        self.gb_chns = gb_chns
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop_prob = drop_prob

    def agnet(self):
        img = Input((1, self.img_size[0], 
                       self.img_size[1], self.img_size[2]))              
        cam = Input((1, self.img_size[0]/8, self.img_size[1]/8, self.img_size[2]/8))
        
        embed_net = self.backbone()
        _, lr_fm = embed_net(img)
        cam_list = []
        for i_chn in range(self.lr_chns):
            cam_list.append(cam)
        cat_cam = Concatenate(axis=1, name='cam_cat')(cam_list)
        gated_lr_fm = multiply([cat_cam, lr_fm], name='gated_lr_fm')
        
        conv1 = Conv3D(filters=self.gb_chns,
                       kernel_size=[3, 3, 3],
                       padding='same',
                       data_format='channels_first',
                       name='global_conv_1')(gated_lr_fm)
        if self.with_bn:
            conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv3D(filters=self.gb_chns / 2,
                       kernel_size=[3, 3, 3],
                       padding='same',
                       data_format='channels_first',
                       name='global_conv_2')(conv1)
        if self.with_bn:
            conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)
        if self.with_dropout:
            conv2 = Dropout(self.drop_prob)(conv2)
        score = AveragePooling3D(pool_size=(self.img_size[0]/8, 
                                            self.img_size[1]/8,
                                            self.img_size[2]/8), 
                              data_format='channels_first', 
                              name='gb_gap')(conv2)
        score = Flatten(name='gb_score')(score)
        
        prob = Dense(units=1, activation='sigmoid', use_bias=False,
                     name='gb_outputs')(score)

        model = Model(inputs=[img, cam], outputs=[prob])
        
        return model
            
    def backbone(self):
        inputs = Input((self.num_chns, self.img_size[0], 
                        self.img_size[1], self.img_size[2]))
                        
        conv1 = Conv3D(filters=self.feature_depth[0], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv1')(inputs)
        if self.with_bn:
            conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)
        
        conv2 = Conv3D(filters=self.feature_depth[1], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv2')(conv1)
        if self.with_bn:
            conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)
        if self.with_dropout:
            conv2 = Dropout(self.drop_prob)(conv2)

        pool1 = Conv3D(filters=self.feature_depth[1], 
                       kernel_size=[2, 2, 2],
                       strides=[2, 2, 2],
                       padding='valid', 
                       data_format='channels_first', 
                       name='pool1')(conv2)
        if self.with_bn:
            pool1 = BatchNormalization(axis=1)(pool1)
        pool1 = Activation('relu')(pool1)
                   
        conv3 = Conv3D(filters=self.feature_depth[2], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv3')(pool1)
        if self.with_bn:
            conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = Activation('relu')(conv3)

        conv4 = Conv3D(filters=self.feature_depth[3], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv4')(conv3)
        if self.with_bn:
            conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = Activation('relu')(conv4)
        if self.with_dropout:
            conv4 = Dropout(self.drop_prob)(conv4)

        pool2 = Conv3D(filters=self.feature_depth[3], 
                       kernel_size=[2, 2, 2],
                       strides=[2, 2, 2],
                       padding='valid', 
                       data_format='channels_first', 
                       name='pool2')(conv4)
        if self.with_bn:
            pool2 = BatchNormalization(axis=1)(pool2)
        pool2 = Activation('relu')(pool2)
            
        conv5 = Conv3D(filters=self.feature_depth[4], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv5')(pool2)
        if self.with_bn:
            conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = Activation('relu')(conv5)

        conv6 = Conv3D(filters=self.feature_depth[5], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first',
                       name='conv6')(conv5)
        if self.with_bn:
            conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = Activation('relu')(conv6)
        if self.with_dropout:
            conv6 = Dropout(self.drop_prob)(conv6)

        pool3 = Conv3D(filters=self.feature_depth[5], 
                       kernel_size=[2, 2, 2],
                       strides=[2, 2, 2],
                       padding='valid', 
                       data_format='channels_first', 
                       name='pool3')(conv6)
        if self.with_bn:
            pool3 = BatchNormalization(axis=1)(pool3)
        pool3 = Activation('relu')(pool3)

        conv7 = Conv3D(filters=self.feature_depth[6], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first',
                       name='conv7')(pool3)
        if self.with_bn:
            conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = Activation('relu')(conv7)

        conv8 = Conv3D(filters=self.feature_depth[7], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first',
                       name='conv8')(conv7)
        if self.with_bn:
            conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = Activation('relu')(conv8)

        transfer = Conv3D(filters=64,
                           kernel_size=[1, 1, 1],
                           padding='same', 
                           data_format='channels_first',
                           name='transfer')(conv8)
        if self.with_bn:
            transfer = BatchNormalization(axis=1)(transfer)
        transfer = Activation('relu')(transfer)
        if self.with_dropout:
            transfer = Dropout(self.drop_prob)(transfer)   
        
        score = AveragePooling3D(pool_size=(self.img_size[0]/8, 
                                            self.img_size[1]/8,
                                            self.img_size[2]/8), 
                              data_format='channels_first', 
                              name='gap')(transfer)
        score = Flatten(name='score')(score)
        
        prob = Dense(units=1, activation='sigmoid', use_bias=False,
                     name='subject_outputs')(score)
        model = Model(inputs=inputs, outputs=[prob, transfer], name='backbone')
        
        return model


class HybNet(object):
    '''
    HybNet consisting of the local, global & fusion branches
    '''
    def __init__(self, 
                 image_size,
                 patch_size,
                 num_patches,
                 pruned_nns,
                 num_chns, 
                 num_outputs, 
                 lb_feature_depth,
                 gb_feature_depth,
                 num_region_features=64,
                 num_subject_features=64,
                 lr_chns=64,
                 gb_chns=64,
                 with_bn=True,
                 with_dropout=True,
                 drop_prob=0.5):
        self.num_chns = num_chns
        self.num_outputs = num_outputs
        self.img_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.nn_mat = pruned_nns
        self.num_regions = np.shape(pruned_nns)[0]
        self.lb_feature_depth = lb_feature_depth
        self.gb_feature_depth = gb_feature_depth
        self.lr_chns = lr_chns
        self.gb_chns = gb_chns
        self.num_region_features = num_region_features
        self.num_subject_features = num_subject_features
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop_prob = drop_prob

    def fusionnet(self):
        lbnet = self.lbnet()
        gbnet = self.gbnet()
        input_pths = []
        for i_input in range(self.num_patches):
            input_name = 'input_pth_{0}'.format(i_input+1)
            input_pths.append(Input((self.num_chns, self.patch_size, 
                        self.patch_size, self.patch_size), 
                        name=input_name))
        img = Input((1, self.img_size[0], 
                    self.img_size[1], self.img_size[2]), name='input_img')              
        cam = Input((1, self.img_size[0]/8, 
                    self.img_size[1]/8, self.img_size[2]/8), name='input_cam')

        lb_features, _ = lbnet(input_pths)
        gb_features, _ = gbnet([img, cam])
        fuse_features = Concatenate(name='fusion', axis=1)([lb_features, gb_features])
        fuse_features = Dense(units=64, activation='relu',
                              name='fusion_fc1')(fuse_features)
        if self.with_dropout:
                fuse_features = Dropout(self.drop_prob)(fuse_features)
        fuse_features = Dense(units=32, activation='relu',
                              name='fusion_fc2')(fuse_features)
        fuse_outputs = Dense(units=1, activation='sigmoid',
                            name='fusion_outputs')(fuse_features)
        model = Model(inputs=input_pths+[img, cam], outputs=fuse_outputs, name='hybnet')
        return model

    def lbnet(self):
        embed_net = self.lb_backbone()
        # inputs
        inputs = []
        for i_input in range(self.num_patches):
            input_name = 'input_{0}'.format(i_input+1)
            inputs.append(Input((self.num_chns, self.patch_size, 
                        self.patch_size, self.patch_size), 
                        name=input_name))
        #+++++++++++++++++++++++++++++#                      
        ##   patch-level processing   #
        #+++++++++++++++++++++++++++++# 
        patch_features_list, patch_probs_list = [], []
        for i_input in range(self.num_patches):
            feature_map, class_prob = embed_net(inputs[i_input])
            patch_features_list.append(feature_map)
            patch_probs_list.append(class_prob)     
        patch_outputs = Concatenate(name='patch_outputs', 
                                    axis=1)(patch_probs_list)                            
        #+++++++++++++++++++++++++++++++#                      
        ##   region-level processing   ## 
        #+++++++++++++++++++++++++++++++#
        region_features_list, region_probs_list = [], []
        for i_region in range(self.num_regions):
            nn_idxs = self.nn_mat[i_region][0][0].tolist()
            nn_features, nn_probs = [], []
            num_neighbors = len(nn_idxs)
            for i_neighbor in nn_idxs:
                nn_features.append(patch_features_list[i_neighbor])
                nn_probs.append(patch_probs_list[i_neighbor])
            region_input_features = Concatenate(axis=1)(nn_features)
            region_input_probs = Concatenate(axis=1)(nn_probs)
            region_input_probs = Reshape([num_neighbors, 1])(region_input_probs)
            in_name = 'region_input_{0}'.format(i_region+1)
            region_input = Concatenate(name=in_name, 
                                       axis=-1)([region_input_probs, region_input_features])
            conv_name = 'region_conv_{0}'.format(i_region+1)
            region_feature = Conv1D(filters=self.num_region_features,
                                    kernel_size=2,
                                    padding='same', 
                                    name=conv_name)(region_input)
            if self.with_bn:
                region_feature = BatchNormalization(axis=-1)(region_feature)
            region_feature = Activation('relu')(region_feature)
            if self.with_dropout:
                region_feature = Dropout(self.drop_prob)(region_feature) 
            gap_name = 'region_gap_{0}'.format(i_region+1)
            region_feature = AveragePooling1D(pool_size=num_neighbors,
                                            name=gap_name)(region_feature)
            ot_name = 'region_prob_{0}'.format(i_region+1)
            region_prob = Dense(units=1, activation='sigmoid',
                                name=ot_name)(Flatten()(region_feature))                      
            region_features_list.append(region_feature)
            region_probs_list.append(region_prob)           
        region_outputs = Concatenate(name='region_outputs', 
                                     axis=1)(region_probs_list)
        region_features = Concatenate(name='region_features', 
                                      axis=1)(region_features_list)
        region_probs = Reshape([self.num_regions, 1], 
                               name='region_probs')(region_outputs) 
        region_feat_prob = Concatenate(name='region_features_probs', 
                                      axis=-1)([region_probs, region_features]) 
        #+++++++++++++++++++++++++++++++#                      
        ##   subject-level processing   #
        #+++++++++++++++++++++++++++++++#
        subject_feature = Conv1D(filters=self.num_subject_features,
                                 kernel_size=2,
                                 padding='same', 
                                 name='subject_conv_1')(region_feat_prob)
        if self.with_bn:
            subject_feature = BatchNormalization(axis=-1)(subject_feature)
        subject_feature = Activation('relu')(subject_feature)

        subject_feature = Conv1D(filters=self.num_subject_features / 2,
                                 kernel_size=2,
                                 padding='same', 
                                 name='subject_conv_2')(region_feat_prob)
        if self.with_bn:
            subject_feature = BatchNormalization(axis=-1)(subject_feature)
        subject_feature = Activation('relu')(subject_feature)

        if self.with_dropout:
            subject_feature = Dropout(self.drop_prob)(subject_feature)
        
        subject_feature = AveragePooling1D(pool_size=self.num_regions,
                                            name='subject_gap')(subject_feature)
        # subject-level units
        subject_units = Flatten(name='subject_level_units')(subject_feature)
        subject_outputs = Dense(units=1, activation='sigmoid',
                                name='subject_outputs')(subject_units)
        outputs = [subject_units, subject_outputs] 
        model = Model(inputs=inputs, outputs=outputs, name='lb')
        return model    
    def lb_backbone(self):
        """ Input with channel first"""
        inputs = Input((self.num_chns, self.patch_size, 
                        self.patch_size, self.patch_size))              
        """ 1st convolution"""                
        conv1 = Conv3D(filters=self.lb_feature_depth[0], 
                       kernel_size=[4, 4, 4],
                       padding='valid', 
                       data_format='channels_first')(inputs)
        if self.with_bn:
            conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)
        """ 2nd convolution"""
        conv2 = Conv3D(filters=self.lb_feature_depth[1], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(conv1)
        if self.with_bn:
            conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)
        """ pooling 1"""
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), 
                             data_format='channels_first')(conv2)
        """ 3rd convolution"""
        conv3 = Conv3D(filters=self.lb_feature_depth[2], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(pool1)
        if self.with_bn:
            conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = Activation('relu')(conv3)
        """ 4th convolution"""
        conv4 = Conv3D(filters=self.lb_feature_depth[3], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(conv3)
        if self.with_bn:
            conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = Activation('relu')(conv4)        
        """ pooling 2"""
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), 
                             data_format='channels_first')(conv4)                             
        """ 5th convolution"""
        conv5 = Conv3D(filters=self.lb_feature_depth[4], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(pool2)
        if self.with_bn:
            conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = Activation('relu')(conv5)
        """ 6th convolution"""
        conv6 = Conv3D(filters=self.lb_feature_depth[5], 
                       kernel_size=[1, 1, 1],
                       padding='valid', 
                       data_format='channels_first')(conv5)
        if self.with_bn:
            conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = Activation('relu')(conv6)
        if self.with_dropout:
            conv6 = Dropout(self.drop_prob)(conv6)    
        feature_map = Reshape((1, self.lb_feature_depth[-1]),
                              name='patch_features')(conv6)    
        class_prob = Dense(units=1, activation='sigmoid',
                           name='patch_prob')(Flatten()(feature_map))        
        model = Model(inputs=inputs, 
                      outputs=[feature_map, class_prob], 
                      name='lb_backbone')
        return model

    def gbnet(self):
        img = Input((1, self.img_size[0], 
                       self.img_size[1], self.img_size[2]))              
        cam = Input((1, self.img_size[0]/8, self.img_size[1]/8, self.img_size[2]/8))
        embed_net = self.gb_backbone()
        _, lr_fm = embed_net(img)
        cam_list = []
        for i_chn in range(self.lr_chns):
            cam_list.append(cam)
        cat_cam = Concatenate(axis=1, name='cam_cat')(cam_list)
        gated_lr_fm = multiply([cat_cam, lr_fm], name='gated_lr_fm')
        conv1 = Conv3D(filters=self.gb_chns,
                       kernel_size=[3, 3, 3],
                       padding='same',
                       data_format='channels_first',
                       name='global_conv_1')(gated_lr_fm)
        if self.with_bn:
            conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv3D(filters=self.gb_chns / 2,
                       kernel_size=[3, 3, 3],
                       padding='same',
                       data_format='channels_first',
                       name='global_conv_2')(conv1)
        if self.with_bn:
            conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)
        if self.with_dropout:
            conv2 = Dropout(self.drop_prob)(conv2)
        score = AveragePooling3D(pool_size=(self.img_size[0]/8, 
                                            self.img_size[1]/8,
                                            self.img_size[2]/8), 
                              data_format='channels_first', 
                              name='gb_gap')(conv2)
        score = Flatten(name='gb_score')(score)
        prob = Dense(units=1, activation='sigmoid', use_bias=False,
                     name='gb_outputs')(score)
        model = Model(inputs=[img, cam], outputs=[score, prob], name='gb')
        return model            
    def gb_backbone(self):
        inputs = Input((self.num_chns, self.img_size[0], 
                        self.img_size[1], self.img_size[2]))                
        conv1 = Conv3D(filters=self.gb_feature_depth[0], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv1')(inputs)
        if self.with_bn:
            conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv3D(filters=self.gb_feature_depth[1], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv2')(conv1)
        if self.with_bn:
            conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)
        if self.with_dropout:
            conv2 = Dropout(self.drop_prob)(conv2)
        pool1 = Conv3D(filters=self.gb_feature_depth[1], 
                       kernel_size=[2, 2, 2],
                       strides=[2, 2, 2],
                       padding='valid', 
                       data_format='channels_first', 
                       name='pool1')(conv2)
        if self.with_bn:
            pool1 = BatchNormalization(axis=1)(pool1)
        pool1 = Activation('relu')(pool1)           
        conv3 = Conv3D(filters=self.gb_feature_depth[2], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv3')(pool1)
        if self.with_bn:
            conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv4 = Conv3D(filters=self.gb_feature_depth[3], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv4')(conv3)
        if self.with_bn:
            conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = Activation('relu')(conv4)
        if self.with_dropout:
            conv4 = Dropout(self.drop_prob)(conv4)
        pool2 = Conv3D(filters=self.gb_feature_depth[3], 
                       kernel_size=[2, 2, 2],
                       strides=[2, 2, 2],
                       padding='valid', 
                       data_format='channels_first', 
                       name='pool2')(conv4)
        if self.with_bn:
            pool2 = BatchNormalization(axis=1)(pool2)
        pool2 = Activation('relu')(pool2)    
        conv5 = Conv3D(filters=self.gb_feature_depth[4], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first', 
                       name='conv5')(pool2)
        if self.with_bn:
            conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = Activation('relu')(conv5)
        conv6 = Conv3D(filters=self.gb_feature_depth[5], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first',
                       name='conv6')(conv5)
        if self.with_bn:
            conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = Activation('relu')(conv6)
        if self.with_dropout:
            conv6 = Dropout(self.drop_prob)(conv6)
        pool3 = Conv3D(filters=self.gb_feature_depth[5], 
                       kernel_size=[2, 2, 2],
                       strides=[2, 2, 2],
                       padding='valid', 
                       data_format='channels_first', 
                       name='pool3')(conv6)
        if self.with_bn:
            pool3 = BatchNormalization(axis=1)(pool3)
        pool3 = Activation('relu')(pool3)
        conv7 = Conv3D(filters=self.gb_feature_depth[6], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first',
                       name='conv7')(pool3)
        if self.with_bn:
            conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = Activation('relu')(conv7)
        conv8 = Conv3D(filters=self.gb_feature_depth[7], 
                       kernel_size=[3, 3, 3],
                       padding='same', 
                       data_format='channels_first',
                       name='conv8')(conv7)
        if self.with_bn:
            conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = Activation('relu')(conv8)
        transfer = Conv3D(filters=64,
                           kernel_size=[1, 1, 1],
                           padding='same', 
                           data_format='channels_first',
                           name='transfer')(conv8)
        if self.with_bn:
            transfer = BatchNormalization(axis=1)(transfer)
        transfer = Activation('relu')(transfer)
        if self.with_dropout:
            transfer = Dropout(self.drop_prob)(transfer)          
        score = AveragePooling3D(pool_size=(self.img_size[0]/8, 
                                            self.img_size[1]/8,
                                            self.img_size[2]/8), 
                              data_format='channels_first', 
                              name='gap')(transfer)
        score = Flatten(name='score')(score)        
        prob = Dense(units=1, activation='sigmoid', use_bias=False,
                     name='subject_outputs')(score)
        model = Model(inputs=inputs, outputs=[prob, transfer], name='gb_backbone')
        return model


if __name__ == '__main__':
    lb_feature_depth = [32, 64, 64, 128, 128, 64]
    gb_feature_depth = [16, 16, 32, 32, 64, 64, 128, 128]
    pruned_nns = sio.loadmat('./files/pru_nns_idx_pthlmk.mat')
    pruned_nns = pruned_nns['pruned_nns_idx']
    pruned_pths = sio.loadmat('files/pru_pths_pthlmk.mat')
    pruned_pths = pruned_pths['pruned_patches']
    pruned_pths = pruned_pths.flatten().tolist()
    num_pths = len(pruned_pths)
    net = HybNet(image_size=[144, 184, 152],
                  patch_size=25,
                  num_patches=num_pths,
                  pruned_nns=pruned_nns,
                  num_chns=1, 
                  num_outputs=1, 
                  lb_feature_depth=lb_feature_depth,
                  gb_feature_depth=gb_feature_depth, 
                  num_region_features=64,
                  num_subject_features=64,
                  lr_chns=64,
                  gb_chns=64,
                  with_bn=True,
                  with_dropout=True,
                  drop_prob=0.5).fusionnet()
    net.summary()