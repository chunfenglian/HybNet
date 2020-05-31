# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:57:03 2017

@author: chlian
"""

import argparse
import keras
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adadelta,Adam
from network import HierachicalNet, acc
from feed_data import data_flow
from loss import binary_cross_entropy, mse

parser = argparse.ArgumentParser(description='Train the H-FCN in the local branch of HybNet.')
parser.add_argument('--with_bn', type=bool, default=True)
parser.add_argument('--with_drop', type=bool, default=True)
parser.add_argument('--drop_prob', type=float, default=0.5)
parser.add_argument('--share_weights', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=100)

parser.add_argument('--img_path', default='/shenlab/lab_stor4/cflian/norm_Crop_AD1_ADNC/')
parser.add_argument('--img_size', default=[144, 184, 152])
parser.add_argument('--save_path', default='./saved_model/')
parser.add_argument('--model_name', default='stage2_hfcn')
parser.add_argument('--gpu', type=int, default=0)

NUM_CHNS = 1
FEATURE_DEPTH = [32, 64, 64, 128, 128, 64]
NUM_REGION_FEATURES = 64
NUM_SUBJECT_FEATURES = 64

CENTER_MAT_NAME = 'top_pths_lmks'
CENTER_MAT = sio.loadmat('files/'+CENTER_MAT_NAME+'.mat')
CENTER_MAT = CENTER_MAT['top_patches'].astype(int)

NUM_PATCHES = np.shape(CENTER_MAT)[1]
NUM_REGION_OUTPUTS = NUM_PATCHES
NUM_SUBJECT_OUTPUTS = 1

NEIGHBOR_MATRIX = sio.loadmat('files/top_pths_lmks_neighbors.mat')
NEIGHBOR_MATRIX = NEIGHBOR_MATRIX['neighbor_matrix'].astype(int) - 1
NUM_NEIGHBORS = 5

DATA_PARTITION = sio.loadmat('files/'+'data_partition.mat')
TRN_SUBJ_LIST = DATA_PARTITION['trn_list'][0].tolist()
TRN_SUBJ_LBLS = DATA_PARTITION['trn_lbls'][0].tolist()
VAL_SUBJ_LIST = DATA_PARTITION['val_list'][0].tolist()
VAL_SUBJ_LBLS = DATA_PARTITION['val_lbls'][0].tolist()

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
      os.makedirs(args.save_path)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    trn_steps = int(np.round(len(TRN_SUBJ_LIST) / args.batch_size))
    val_steps = int(np.round(len(VAL_SUBJ_LIST) / args.batch_size))

    print('Trianing {0} iterations in each epoch'.format(trn_steps))

    Net = HierachicalNet(patch_size=args.patch_size,
                         num_patches=NUM_PATCHES,
                         num_chns=NUM_CHNS, 
                         num_outputs=NUM_SUBJECT_OUTPUTS,
                         feature_depth=FEATURE_DEPTH,
                         num_neighbors=NUM_NEIGHBORS,
                         neighbor_matrix=NEIGHBOR_MATRIX,
                         num_region_features=NUM_REGION_FEATURES,
                         num_subject_features=NUM_SUBJECT_FEATURES,
                         with_bn=args.with_bn,
                         with_dropout=args.with_drop,
                         drop_prob=args.drop_prob,
                         region_sparse=0.0,
                         subject_sparse=0.0).get_global_net()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    Net.compile(loss=binary_cross_entropy, loss_weights=[1.0, 1.0, 1.0], 
                optimizer=adam, metrics=[acc])
    Net.summary()
                
    trn_flow = data_flow(img_path=args.img_path, 
                           sample_list=TRN_SUBJ_LIST, 
                           sample_labels=TRN_SUBJ_LBLS, 
                           center_cors=CENTER_MAT, 
                           batch_size=args.batch_size, 
                           patch_size=args.patch_size, 
                           num_chns=NUM_CHNS, 
                           num_patches=NUM_PATCHES,
                           num_region_outputs=NUM_REGION_OUTPUTS, 
                           num_subject_outputs=NUM_SUBJECT_OUTPUTS,
                           shuffle_flag=True, shift_flag=True,
                           scale_flag=True, flip_flag=True,
                           scale_range=[0.95, 1.05], flip_axis=0,
                           shift_range=[-2, -1, 0, 1, 2])
    val_flow = data_flow(img_path=args.img_path, 
                           sample_list=VAL_SUBJ_LIST, 
                           sample_labels=VAL_SUBJ_LBLS, 
                           center_cors=CENTER_MAT, 
                           batch_size=args.batch_size, 
                           patch_size=args.patch_size, 
                           num_chns=NUM_CHNS, 
                           num_patches=NUM_PATCHES,
                           num_region_outputs=NUM_REGION_OUTPUTS, 
                           num_subject_outputs=NUM_SUBJECT_OUTPUTS,
                           shuffle_flag=False, shift_flag=False, 
                           scale_flag=False, flip_flag=False)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
                      filepath=args.save_path + args.model_name + '.best.h5',
                      save_weights_only=True,
                      monitor='val_subject_outputs_acc', mode='max',
                      save_best_only=True)
    logger = keras.callbacks.CSVLogger(args.save_path + args.model_name + '.log', 
                                      separator=",", append=False)
    
    # TRAIN max_queue_size=10
    Net.fit_generator(generator=trn_flow,
                      steps_per_epoch=trn_steps,
                      epochs=args.num_epochs,
                      validation_data=val_flow,
                      validation_steps=val_steps,
                      callbacks=[checkpoint, logger])