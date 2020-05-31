# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:30:47 2017

@author: chlian
"""

import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio
import itertools
import random

class SequentialLoader(object):
    '''
    data generator for backbone FCN
    '''
    def __init__(self, image_path, 
                 batch_size,
                 num_input_chns,
                 num_outputs,
                 image_size=[144, 184, 152],
                 scale_range=[0.95, 1.05],
                 flip_axis=0,
                 shuffle_flag=True):
        self.img_path = image_path
        self.img_size = image_size
        
        self.batch_size = batch_size
        
        self.num_chns = num_input_chns
        self.num_outputs = num_outputs

        self.shuffle_flag = shuffle_flag
        
        if flip_axis < 0 or flip_axis > 2:
            raise ValueError('flip axis should be 0 -> x, 1 -> y, or 2 -> z.')   
        self.flip_axis = flip_axis 

        self.scale_range = scale_range
        
    def data_flow(self, trn_list, trn_labels, scale_flag=False, flip_flag= False, noise_flag=False):

        input_shape = (self.batch_size, self.num_chns, self.img_size[0],
                       self.img_size[1], self.img_size[2])           
        output_shape = (self.batch_size, self.num_outputs)

        while True:
            
            if self.shuffle_flag:
                trn_list = np.array(trn_list)
                trn_labels = np.array(trn_labels)
                permut = np.random.permutation(len(trn_list))
                np.take(trn_list, permut, out=trn_list)
                np.take(trn_labels, permut, out=trn_labels)
                trn_list = trn_list.tolist()
                trn_labels = trn_labels.tolist()
            
            inputs = np.zeros(input_shape, dtype='float32')
            outputs = np.ones(output_shape, dtype='int8')
            
            i_batch = 0
            for i_iter in range(len(trn_list)):
                
                i_subject = trn_list[i_iter]
                
                img_dir = self.img_path + 'Crop_Img_{0}.nii.gz'
                img = np.array(sitk.GetArrayFromImage(
                                sitk.ReadImage(img_dir.format(i_subject))))

                if flip_flag:
                    flip_action = np.random.randint(0, 2)
                else:
                    flip_action = 0
                if flip_action == 1:
                    if self.flip_axis == 0:
                        img = img[:, :, ::-1]
                    elif self.flip_axis == 1:
                        img = img[:, ::-1, :]
                    elif self.flip_axis ==2:
                        img = img[::-1, :, :]
            
                if noise_flag:
                    noise_action = np.random.randint(0, 2)
                else:
                    noise_action = 0
            
                if noise_action == 1:
                    img = img + np.random.normal(0.0, 2.0, img.shape)
            
                if scale_flag:
                    scale = np.random.uniform(self.scale_range[0],
                                              self.scale_range[1], 3)
                    simg = nd.interpolation.zoom(img, scale, order=1)
                    img = np.zeros((simg.shape[0]+50,
                                    simg.shape[1]+50,
                                    simg.shape[2]+50), dtype='float32')
                    img[:simg.shape[0], :simg.shape[1], :simg.shape[2]] = simg
                    img = img[:self.img_size[0], :self.img_size[1], :self.img_size[2]] 
                    
                inputs[i_batch, 0, :, :, :] = img

                outputs[i_batch, :] = trn_labels[i_iter] * outputs[i_batch, :]
                    
                i_batch += 1
                
                if i_batch == self.batch_size:
                    yield(inputs, outputs)
                    inputs = np.zeros(input_shape, dtype='float32')
                    outputs = np.ones(output_shape, dtype='int8')
                    i_batch = 0


def data_flow(img_path, sample_list, sample_labels, center_cors, 
              batch_size, patch_size, num_chns, num_patches,
              num_region_outputs, num_subject_outputs,
              shuffle_flag=True, shift_flag=True, scale_flag=False, 
              flip_flag=False, scale_range=[0.95, 1.05], flip_axis=0, 
              shift_range=[-2, -1, 0, 1, 2]):
    
    if flip_axis < 0 or flip_axis > 2:
            raise ValueError('flip axis should be 0 -> x, 1 -> y, or 2 -> z.')
            
    margin = int(np.floor((patch_size-1)/2.0))
    
    input_shape = (batch_size, num_chns, patch_size, patch_size, patch_size)
    low_ot_shape = (batch_size, num_patches)
    mid_ot_shape = (batch_size, num_region_outputs)
    high_ot_shape = (batch_size, num_subject_outputs)
    
    while True:
        if shuffle_flag:
            sample_list = np.array(sample_list)
            sample_labels = np.array(sample_labels)
            permut = np.random.permutation(len(sample_list))
            np.take(sample_list, permut, out=sample_list)
            np.take(sample_labels, permut, out=sample_labels)
            sample_list = sample_list.tolist()
            sample_labels = sample_labels.tolist()
            
        inputs = []
        for i_input in range(num_patches):
            inputs.append(np.zeros(input_shape, dtype='float32'))
        outputs = [np.ones(low_ot_shape, dtype='int8'), 
                   np.ones(mid_ot_shape, dtype='int8'),
                   np.ones(high_ot_shape, dtype='int8')]
                   
        i_batch = 0
        for i_iter in range(len(sample_list)):
            # random flip
            if flip_flag:
                flip_action = np.random.randint(0, 2)
            else:
                flip_action = 0
                
            i_subject = sample_list[i_iter]
                
            img_dir = img_path + 'Crop_Img_{0}.nii.gz'
            I = sitk.ReadImage(img_dir.format(i_subject))
            img = np.array(sitk.GetArrayFromImage(I))
            
            img = np.pad(img, ((17,17),(17,17),(17,17)), 'constant')
    
            # random rescale
            if scale_flag:
                scale = np.random.uniform(scale_range[0], scale_range[1], 3)
                img = nd.interpolation.zoom(img, scale, order=1)
                
            center_z = center_cors[0,:].tolist()
            center_y = center_cors[1,:].tolist()
            center_x = center_cors[2,:].tolist()
            for i_patch in range(num_patches):
                x_cor = center_x[i_patch]
                y_cor = center_y[i_patch]
                z_cor = center_z[i_patch]
                    
                if shift_flag:
                    x_scor = x_cor + np.random.choice(shift_range)
                    y_scor = y_cor + np.random.choice(shift_range)
                    z_scor = z_cor + np.random.choice(shift_range)
                else:
                    x_scor, y_scor, z_scor = x_cor, y_cor, z_cor
                
                z_scor += 17
                y_scor += 17
                x_scor += 17
                
                img_patch = img[z_scor-margin: z_scor+margin+1, 
                                y_scor-margin: y_scor+margin+1, 
                                x_scor-margin: x_scor+margin+1]
                                    
                if flip_action == 1:
                    if flip_axis == 0:
                        img_patch = img_patch[:, :, ::-1]
                    elif flip_axis == 1:
                        img_patch = img_patch[:, ::-1, :]
                    elif flip_axis == 2:
                        img_patch = img_patch[::-1, :, :]

                inputs[i_patch][i_batch, 0, :, :, :] = img_patch
                    
            outputs[0][i_batch, :] = \
                    sample_labels[i_iter] * outputs[0][i_batch, :]
            outputs[1][i_batch, :] = \
                    sample_labels[i_iter] * outputs[1][i_batch, :]
            outputs[2][i_batch, :] = \
                    sample_labels[i_iter] * outputs[2][i_batch, :]
                    
            i_batch += 1
                
            if i_batch == batch_size:  
                yield(inputs, outputs)  
                inputs = []
                for i_input in range(num_patches):
                    inputs.append(np.zeros(input_shape, dtype='float32'))
                outputs = [np.ones(low_ot_shape, dtype='int8'), 
                           np.ones(mid_ot_shape, dtype='int8'),
                           np.ones(high_ot_shape, dtype='int8')]
                i_batch = 0                

def tst_data_flow(img_path, sample_idx, sample_lbl, center_cors, 
                  patch_size, num_chns, num_patches,
                  num_region_outputs, num_subject_outputs):

    input_shape = (1, num_chns, patch_size, patch_size, patch_size)
    low_ot_shape = (1, num_patches)
    mid_ot_shape = (1, num_region_outputs)
    high_ot_shape = (1, num_subject_outputs)

    margin = int(np.floor((patch_size-1)/2.0))

    img_dir = img_path + 'Crop_Img_{0}.hdr'
    I = sitk.ReadImage(img_dir.format(sample_idx))
    img = np.array(sitk.GetArrayFromImage(I))
    
    img = np.pad(img, ((17,17),(17,17),(17,17)), 'constant')

    inputs = []
    for i_input in range(num_patches):
        inputs.append(np.zeros(input_shape, dtype='float32'))
    
    center_z = center_cors[0,:].tolist()
    center_y = center_cors[1,:].tolist()
    center_x = center_cors[2,:].tolist()
    for i_patch in range(num_patches):
        x_cor = center_x[i_patch]
        y_cor = center_y[i_patch]
        z_cor = center_z[i_patch]
        
        z_cor += 17
        y_cor += 17
        x_cor += 17
        img_patch = img[z_cor-margin: z_cor+margin+1,
                        y_cor-margin: y_cor+margin+1, 
                        x_cor-margin: x_cor+margin+1]
        inputs[i_patch][0, 0, :, :, :] = img_patch

    outputs = [sample_lbl * np.ones(low_ot_shape, dtype='float32'),
               sample_lbl * np.ones(mid_ot_shape, dtype='float32'),
               sample_lbl * np.ones(high_ot_shape, dtype='float32')]

    return inputs, outputs

def pruned_data_flow(img_path, sample_list, sample_labels, 
              center_cors, patch_idxs, batch_size, 
              patch_size, num_chns, num_patches,
              num_region_outputs, num_subject_outputs,
              shuffle_flag=True, shift_flag=True, scale_flag=False, 
              flip_flag=False, scale_range=[0.95, 1.05], flip_axis=0, 
              shift_range=[-2, -1, 0, 1, 2]):
    
    if flip_axis < 0 or flip_axis > 2:
            raise ValueError('flip axis should be 0 -> x, 1 -> y, or 2 -> z.')
            
    margin = int(np.floor((patch_size-1)/2.0))
    
    input_shape = (batch_size, num_chns, patch_size, patch_size, patch_size)
    low_ot_shape = (batch_size, num_patches)
    mid_ot_shape = (batch_size, num_region_outputs)
    high_ot_shape = (batch_size, num_subject_outputs)
    
    while True:
        if shuffle_flag:
            sample_list = np.array(sample_list)
            sample_labels = np.array(sample_labels)
            permut = np.random.permutation(len(sample_list))
            np.take(sample_list, permut, out=sample_list)
            np.take(sample_labels, permut, out=sample_labels)
            sample_list = sample_list.tolist()
            sample_labels = sample_labels.tolist()
            
        inputs = []
        for i_input in range(num_patches):
            inputs.append(np.zeros(input_shape, dtype='float32'))
        outputs = [np.ones(low_ot_shape, dtype='int8'), 
                   np.ones(mid_ot_shape, dtype='int8'),
                   np.ones(high_ot_shape, dtype='int8')]
                   
        i_batch = 0
        for i_iter in range(len(sample_list)):
            # random flip
            if flip_flag:
                flip_action = np.random.randint(0, 2)
            else:
                flip_action = 0
                
            i_subject = sample_list[i_iter]
                
            img_dir = img_path + 'Crop_Img_{0}.hdr'
            img = np.array(sitk.GetArrayFromImage(sitk.ReadImage(img_dir.format(i_subject))))
            
            img = np.pad(img, ((17,17),(17,17),(17,17)), 'constant')
    
            # random rescale
            if scale_flag:
                scale = np.random.uniform(scale_range[0], scale_range[1], 3)
                img = nd.interpolation.zoom(img, scale, order=1)
                
            center_z = center_cors[0,:].tolist()
            center_y = center_cors[1,:].tolist()
            center_x = center_cors[2,:].tolist()
            for i_patch in range(num_patches):
                x_cor = center_x[patch_idxs[i_patch]]
                y_cor = center_y[patch_idxs[i_patch]]
                z_cor = center_z[patch_idxs[i_patch]]
                    
                if shift_flag:
                    x_scor = x_cor + np.random.choice(shift_range)
                    y_scor = y_cor + np.random.choice(shift_range)
                    z_scor = z_cor + np.random.choice(shift_range)
                else:
                    x_scor, y_scor, z_scor = x_cor, y_cor, z_cor
                
                z_sscor = x_scor + 17
                y_sscor = y_scor + 17
                x_sscor = z_scor + 17
                
                img_patch = img[z_sscor-margin: z_sscor+margin+1,
                                y_sscor-margin: y_sscor+margin+1,
                                x_sscor-margin: x_sscor+margin+1]
                                    
                if flip_action == 1:
                    if flip_axis == 0:
                        img_patch = img_patch[:, :, ::-1]
                    elif flip_axis == 1:
                        img_patch = img_patch[:, ::-1, :]
                    elif flip_axis == 2:
                        img_patch = img_patch[::-1, :, :]

                inputs[i_patch][i_batch, 0, :, :, :] = img_patch
                    
            outputs[0][i_batch, :] = \
                    sample_labels[i_iter] * outputs[0][i_batch, :]
            outputs[1][i_batch, :] = \
                    sample_labels[i_iter] * outputs[1][i_batch, :]
            outputs[2][i_batch, :] = \
                    sample_labels[i_iter] * outputs[2][i_batch, :]
                    
            i_batch += 1
                
            if i_batch == batch_size:  
                yield(inputs, outputs)  
                inputs = []
                for i_input in range(num_patches):
                    inputs.append(np.zeros(input_shape, dtype='float32'))
                outputs = [np.ones(low_ot_shape, dtype='int8'), 
                           np.ones(mid_ot_shape, dtype='int8'),
                           np.ones(high_ot_shape, dtype='int8')]
                i_batch = 0

class Stage2GlobalLoader(object):
    def __init__(self, image_path,
                 cam_path,
                 batch_size,
                 num_input_chns,
                 num_outputs,
                 image_size=[144, 184, 152],
                 scale_range=[0.95, 1.05],
                 flip_axis=0,
                 shuffle_flag=True):
        self.img_path = image_path
        self.img_size = image_size
        self.cam_path = cam_path
        self.batch_size = batch_size
        self.num_chns = num_input_chns
        self.num_outputs = num_outputs
        self.shuffle_flag = shuffle_flag
        
        if flip_axis < 0 or flip_axis > 2:
            raise ValueError('flip axis should be 0 -> x, 1 -> y, or 2 -> z.')   
        self.flip_axis = flip_axis 

        self.scale_range = scale_range
        
    def data_flow(self, trn_list, trn_labels, flip_flag=False, noise_flag=False):

        input_shape = (self.batch_size, self.num_chns, self.img_size[0],
                       self.img_size[1], self.img_size[2])  
        cam_shape = (self.batch_size, self.num_chns, self.img_size[0] / 8,
                     self.img_size[1] / 8, self.img_size[2] / 8)
        output_shape = (self.batch_size, self.num_outputs)
        
        while True: 
            if self.shuffle_flag:
                trn_list = np.array(trn_list)
                trn_labels = np.array(trn_labels)
                permut = np.random.permutation(len(trn_list))
                np.take(trn_list, permut, out=trn_list)
                np.take(trn_labels, permut, out=trn_labels)
                trn_list = trn_list.tolist()
                trn_labels = trn_labels.tolist()
            
            inputs = np.zeros(input_shape, dtype='float32')
            cams = np.zeros(cam_shape, dtype='float32')
            outputs = np.ones(output_shape, dtype='int8')
            
            i_batch = 0
            for i_iter in range(len(trn_list)):
                
                i_subject = trn_list[i_iter]
                img_dir = self.img_path + 'Crop_Img_{0}.nii.gz'
                img = 1.0 * np.array(sitk.GetArrayFromImage(
                                sitk.ReadImage(img_dir.format(i_subject))))
                
                cam_dir = self.cam_path + 'Raw_CAM_Subj{0}.nii.gz'
                cam = np.array(sitk.GetArrayFromImage(
                                sitk.ReadImage(cam_dir.format(i_subject))))
                
                if flip_flag:
                    flip_action = np.random.randint(0, 2)
                else:
                    flip_action = 0
                if flip_action == 1:
                    if self.flip_axis == 0:
                        img = img[:, :, ::-1]
                        cam = cam[:, :, ::-1]
                    elif self.flip_axis == 1:
                        img = img[:, ::-1, :]
                        cam = cam[:, ::-1, :]
                    elif self.flip_axis ==2:
                        img = img[::-1, :, :]
                        cam = cam[::-1, :, :]
            
                if noise_flag:
                    noise_action = np.random.randint(0, 2)
                else:
                    noise_action = 0 
                if noise_action == 1:
                    img += np.random.normal(0, 2, img.shape)
                    cam += np.random.uniform(0, 0.2, cam.shape)

                inputs[i_batch, 0, :, :, :] = img
                cams[i_batch, 0, :, :, :] = cam
                outputs[i_batch, :] = trn_labels[i_iter] * outputs[i_batch, :]
                
                i_batch += 1
                
                if i_batch == self.batch_size:
                    yield([inputs, cams], outputs)
                    inputs = np.zeros(input_shape, dtype='float32')
                    cams = np.zeros(cam_shape, dtype='float32')
                    outputs = np.ones(output_shape, dtype='int8')
                    i_batch = 0


class Stage2HybNetLoader(object):
    def __init__(self, image_path, cam_path,
                 center_cors, patch_idxs, 
                 batch_size,
                 patch_size, num_patches,
                 num_input_chns,
                 num_outputs,
                 image_size=[144, 184, 152],
                 scale_range=[0.95, 1.05],
                 flip_axis=0,
                 shift_range=[-2, -1, 0, 1, 2],
                 shuffle_flag=True):
        self.img_path = image_path
        self.img_size = image_size
        self.cam_path = cam_path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_chns = num_input_chns
        self.num_outputs = num_outputs
        self.num_patches = num_patches
        self.center_cors = center_cors
        self.patch_idxs = patch_idxs
        self.shuffle_flag = shuffle_flag
        self.shift_range = shift_range
        if flip_axis < 0 or flip_axis > 2:
            raise ValueError('flip axis should be 0 -> x, 1 -> y, or 2 -> z.')   
        self.flip_axis = flip_axis 
        self.scale_range = scale_range
        
    def data_flow(self, trn_list, trn_labels, flip_flag=False, shift_flag=False, noise_flag=False):
        margin = int(np.floor((self.patch_size - 1) / 2.0))
        patch_shape = (self.batch_size, self.num_chns) + (self.patch_size,) * 3
        input_shape = (self.batch_size, self.num_chns, self.img_size[0],
                       self.img_size[1], self.img_size[2])  
        cam_shape = (self.batch_size, self.num_chns, self.img_size[0] / 8,
                     self.img_size[1] / 8, self.img_size[2] / 8)
        output_shape = (self.batch_size, self.num_outputs)
        
        while True: 
            if self.shuffle_flag:
                trn_list = np.array(trn_list)
                trn_labels = np.array(trn_labels)
                permut = np.random.permutation(len(trn_list))
                np.take(trn_list, permut, out=trn_list)
                np.take(trn_labels, permut, out=trn_labels)
                trn_list = trn_list.tolist()
                trn_labels = trn_labels.tolist()

            patches = []
            for i_input in range(self.num_patches):
                patches.append(np.zeros(patch_shape, dtype='float32'))
            inputs = np.zeros(input_shape, dtype='float32')
            cams = np.zeros(cam_shape, dtype='float32')
            outputs = np.ones(output_shape, dtype='int8')
            
            i_batch = 0
            for i_iter in range(len(trn_list)):   
                i_subject = trn_list[i_iter]
                img_dir = self.img_path + 'Crop_Img_{0}.nii.gz'
                img = 1.0 * np.array(sitk.GetArrayFromImage(
                                sitk.ReadImage(img_dir.format(i_subject))))
                cam_dir = self.cam_path + 'Raw_CAM_Subj{0}.nii.gz'
                cam = np.array(sitk.GetArrayFromImage(
                                sitk.ReadImage(cam_dir.format(i_subject))))
                pimg = np.pad(img, ((17,17),(17,17),(17,17)), 'constant')

                if flip_flag:
                    flip_action = np.random.randint(0, 2)
                else:
                    flip_action = 0
                if flip_action == 1:
                    if self.flip_axis == 0:
                        img = img[:, :, ::-1]
                        cam = cam[:, :, ::-1]
                    elif self.flip_axis == 1:
                        img = img[:, ::-1, :]
                        cam = cam[:, ::-1, :]
                    elif self.flip_axis ==2:
                        img = img[::-1, :, :]
                        cam = cam[::-1, :, :]
                if noise_flag:
                    noise_action = np.random.randint(0, 2)
                else:
                    noise_action = 0 
                if noise_action == 1:
                    img += np.random.normal(0, 2, img.shape)
                    pimg = np.pad(img, ((17,17),(17,17),(17,17)), 'constant') 
                    cam += np.random.uniform(0, 0.2, cam.shape)

                center_z = self.center_cors[0,:].tolist()
                center_y = self.center_cors[1,:].tolist()
                center_x = self.center_cors[2,:].tolist()
                for i_patch in range(self.num_patches):
                    x_cor = center_x[self.patch_idxs[i_patch]]
                    y_cor = center_y[self.patch_idxs[i_patch]]
                    z_cor = center_z[self.patch_idxs[i_patch]]
                    if shift_flag:
                        x_scor = x_cor + np.random.choice(self.shift_range)
                        y_scor = y_cor + np.random.choice(self.shift_range)
                        z_scor = z_cor + np.random.choice(self.shift_range)
                    else:
                        x_scor, y_scor, z_scor = x_cor, y_cor, z_cor
                    z_sscor = x_scor + 17
                    y_sscor = y_scor + 17
                    x_sscor = z_scor + 17
                    img_patch = pimg[z_sscor-margin: z_sscor+margin+1,
                                    y_sscor-margin: y_sscor+margin+1,
                                    x_sscor-margin: x_sscor+margin+1]                  
                    if flip_action == 1:
                        if self.flip_axis == 0:
                            img_patch = img_patch[:, :, ::-1]
                        elif self.flip_axis == 1:
                            img_patch = img_patch[:, ::-1, :]
                        elif self.flip_axis == 2:
                            img_patch = img_patch[::-1, :, :]
                    patches[i_patch][i_batch, 0, :, :, :] = img_patch

                inputs[i_batch, 0, :, :, :] = img
                cams[i_batch, 0, :, :, :] = cam
                outputs[i_batch, :] = trn_labels[i_iter] * outputs[i_batch, :]
                
                i_batch += 1
                
                if i_batch == self.batch_size:
                    yield(patches + [inputs, cams], outputs)
                    patches = []
                    for i_input in range(self.num_patches):
                        patches.append(np.zeros(patch_shape, dtype='float32'))
                    inputs = np.zeros(input_shape, dtype='float32')
                    cams = np.zeros(cam_shape, dtype='float32')
                    outputs = np.ones(output_shape, dtype='int8')
                    i_batch = 0
