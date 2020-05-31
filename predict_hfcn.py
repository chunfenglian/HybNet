# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:52:35 2017

@author: chlian
"""

import keras
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio
from keras.models import Model
from network import HierachicalNet, acc
from feed_data import tst_data_flow
from loss import binary_cross_entropy, mse
import argparse

parser = argparse.ArgumentParser(description='Apply the initally trained H-FCN to inference.')
parser.add_argument('--with_bn', type=bool, default=True)
parser.add_argument('--with_drop', type=bool, default=True)
parser.add_argument('--drop_prob', type=float, default=0.5)
parser.add_argument('--share_weights', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=100)

parser.add_argument('--img_path', default='/shenlab/lab_stor4/cflian/norm_Crop_AD1_ADNC/')
parser.add_argument('--img_size', default=[144, 184, 152])
parser.add_argument('--resume', default='./saved_model/stage2_hfcn_pretrain.best.hd5')
parser.add_argument('--save_path', default='./saved_results/')
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
    Net.compile(loss=binary_cross_entropy, loss_weights=[1.0, 1.0, 1.0],
                optimizer='Adam', metrics=[acc])                     
    Net.load_weights(args.resume)
    
    patch_outputs = np.zeros((len(VAL_SUBJ_LIST), NUM_PATCHES))
    patch_scores = np.zeros((len(VAL_SUBJ_LIST), NUM_PATCHES))
    region_outputs = np.zeros((len(VAL_SUBJ_LIST), NUM_REGION_OUTPUTS))
    region_scores = np.zeros((len(VAL_SUBJ_LIST), NUM_REGION_OUTPUTS))
    subject_outputs = np.zeros((len(VAL_SUBJ_LIST), NUM_SUBJECT_OUTPUTS))
    subject_scores = np.zeros((len(VAL_SUBJ_LIST), NUM_SUBJECT_OUTPUTS))                     
    for i in range(len(VAL_SUBJ_LIST)):
        inputs, outputs = tst_data_flow(img_path=args.img_path, 
                           sample_idx=VAL_SUBJ_LIST[i], 
                           sample_lbl=VAL_SUBJ_LBLS[i], 
                           center_cors=CENTER_MAT, 
                           patch_size=args.patch_size, 
                           num_chns=NUM_CHNS, 
                           num_patches=NUM_PATCHES,
                           num_region_outputs=NUM_REGION_OUTPUTS, 
                           num_subject_outputs=NUM_SUBJECT_OUTPUTS)                            
        predicts = Net.predict(inputs)

        patch_outputs[i, :] = predicts[0]
        region_outputs[i, :] = predicts[1]
        subject_outputs[i, :] = predicts[2]
        patch_scores[i, :] = 1.0 - np.abs(outputs[0] - predicts[0])
        region_scores[i, :] = 1.0 - np.abs(outputs[1] - predicts[1])
        subject_scores[i, :] = 1.0 - np.abs(outputs[2] - predicts[2]) 
        print i

    sio.savemat(args.save_path + 'results_stage2_hfcn.mat',
                {'patch_outputs': patch_outputs, 'patch_scores': patch_scores, 
                'region_outputs': region_outputs, 'region_scores': region_scores, 
                'subject_outputs': subject_outputs, 'subject_scores': subject_scores})

