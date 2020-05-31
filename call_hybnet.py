# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:09:22 2018

@author: chlian
"""

import keras
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio
from keras.models import Model
from network import HybNet
from feed_data import Stage2HybNetLoader
import argparse

parser = argparse.ArgumentParser(description='Train HybNet for AD diagnosis.')
parser.add_argument('--with_bn', type=bool, default=True)
parser.add_argument('--with_drop', type=bool, default=True)
parser.add_argument('--drop_prob', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_outputs', type=int, default=1)

parser.add_argument('--img_path', default='/shenlab/lab_stor4/cflian/norm_Crop_AD1_ADNC/')
parser.add_argument('--cam_path', default='/shenlab/lab_stor4/cflian/cam_ad1_adnc/')
parser.add_argument('--img_size', default=[144, 184, 152])
parser.add_argument('--patch_size', type=int, default=25)
parser.add_argument('--save_path', default='./saved_model/')
parser.add_argument('--model_name', default='stage2_hybnet')
parser.add_argument('--resume', default='./saved_model/stage2_hybnet.best.h5')
parser.add_argument('--gpu', type=int, default=0)

NUM_GB_FEATURES = 64

GB_FEATURE_DEPTH = [16, 16, 32, 32, 64, 64, 128, 128]
LB_FEATURE_DEPTH = [32, 64, 64, 128, 128, 64]
NUM_REGION_FEATURES = 64
NUM_SUBJECT_FEATURES = 64

CENTER_MAT_NAME = 'top_pths_lmks'
CENTER_MAT = sio.loadmat('files/'+CENTER_MAT_NAME+'.mat')
CENTER_MAT = CENTER_MAT['top_patches']
CENTER_MAT = np.round(CENTER_MAT).astype(int)

PRUNED_NEIGHBORS = sio.loadmat('files/pru_nns_idx_pthlmk.mat')
PRUNED_NEIGHBORS = PRUNED_NEIGHBORS['pruned_nns_idx']

PRUNED_PATCHES = sio.loadmat('files/pru_pths_pthlmk.mat')
PRUNED_PATCHES = PRUNED_PATCHES['pruned_patches']
PRUNED_PATCHES = PRUNED_PATCHES.flatten().tolist()
NUM_PATCHES = len(PRUNED_PATCHES)

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

    Net = HybNet(image_size=args.img_size,
                  patch_size=args.patch_size,
                  num_patches=NUM_PATCHES,
                  pruned_nns=PRUNED_NEIGHBORS,
                  num_chns=1, 
                  num_outputs=1, 
                  lb_feature_depth=LB_FEATURE_DEPTH,
                  gb_feature_depth=GB_FEATURE_DEPTH, 
                  num_region_features=NUM_REGION_FEATURES,
                  num_subject_features=NUM_SUBJECT_FEATURES,
                  lr_chns=NUM_GB_FEATURES,
                  gb_chns=NUM_GB_FEATURES,
                  with_bn=args.with_bn, with_dropout=args.with_drop, 
                  drop_prob=args.drop_prob).fusionnet()
    Net.get_layer('gb').load_weights(args.save_path + 'stage2_attgatednet.best.h5')
    Net.get_layer('lb').load_weights(args.save_path + 'stage2_pruned_hfcn.best.h5')
    Net.get_layer('gb').trainable = False
    Net.get_layer('lb').trainable = False
    Net.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    Net.summary()

    generator = Stage2HybNetLoader(image_path=args.img_path,
                                  cam_path=args.cam_path, 
                                  center_cors=CENTER_MAT, 
                                  patch_idxs=PRUNED_PATCHES,
                                  batch_size=args.batch_size,
                                  patch_size=args.patch_size,
                                  num_patches=NUM_PATCHES,
                                  num_input_chns=1,
                                  num_outputs=args.num_outputs,
                                  image_size=args.img_size,
                                  scale_range=[0.95, 1.05],
                                  flip_axis=0,
                                  shift_range=[-2, -1, 0, 1, 2],
                                  shuffle_flag=True)
    train_flow = generator.data_flow(TRN_SUBJ_LIST, TRN_SUBJ_LBLS, 
                                     flip_flag=True, shift_flag=True, noise_flag=True)
    validate_flow = generator.data_flow(VAL_SUBJ_LIST, VAL_SUBJ_LBLS, 
                                        flip_flag=False, shift_flag=False, noise_flag=False)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
                      filepath=args.save_path + args.model_name + '.best.h5',
                      save_weights_only=True,
                      monitor='val_accuracy', mode='max', save_best_only=True)
    logger = keras.callbacks.CSVLogger(args.save_path + args.model_name + '.log', 
                                      separator=",", append=False)

    Net.fit_generator(generator=train_flow,
                      steps_per_epoch=trn_steps,
                      epochs=args.num_epochs,
                      validation_data=validate_flow,
                      validation_steps=val_steps,
                      callbacks=[checkpoint, logger])


    '''
    NOTE: latest Keras versions may have the bug 
          in loading the saved weights for nested models.
    '''
    #Net.load_weights(args.resume)
    cls_score = []
    for i in range(len(VAL_SUBJ_LIST)):
        
        i_subj = TRN_SUBJ_LIST[i]
        print args.img_path + 'Crop_Img_{0}.nii.gz'.format(i_subj)
        
        img_dir = args.img_path + 'Crop_Img_{0}.nii.gz'
        img = 1.0 * np.array(sitk.GetArrayFromImage(sitk.ReadImage(img_dir.format(i_subj))))
        cam_dir = args.cam_path + 'Raw_CAM_Subj{0}.nii.gz'
        cam = np.array(sitk.GetArrayFromImage(
                        sitk.ReadImage(cam_dir.format(i_subj))))
        pimg = np.pad(img, ((17,17),(17,17),(17,17)), 'constant')

        margin = int(np.floor((args.patch_size - 1) / 2.0))
        patch_shape = (1, 1) + (args.patch_size,) * 3
        input_shape = [1, 1] + args.img_size
        cam_shape = (1, 1, args.img_size[0] / 8, args.img_size[1] / 8, args.img_size[2] / 8)
        center_z = CENTER_MAT[0,:].tolist()
        center_y = CENTER_MAT[1,:].tolist()
        center_x = CENTER_MAT[2,:].tolist()

        patches = []
        for i_input in range(NUM_PATCHES):
            patches.append(np.zeros(patch_shape, dtype='float32'))
        inputs = np.zeros(input_shape, dtype='float32')
        cams = np.zeros(cam_shape, dtype='float32')

        inputs[0, 0, :, :, :] = img
        cams[0, 0, :, :, :] = cam
        for i_patch in range(NUM_PATCHES):
            x_cor = center_x[PRUNED_PATCHES[i_patch]]
            y_cor = center_y[PRUNED_PATCHES[i_patch]]
            z_cor = center_z[PRUNED_PATCHES[i_patch]]
            z_sscor = x_cor + 17
            y_sscor = y_cor + 17
            x_sscor = z_cor + 17
            img_patch = pimg[z_sscor-margin: z_sscor+margin+1,
                            y_sscor-margin: y_sscor+margin+1,
                            x_sscor-margin: x_sscor+margin+1]                  
            patches[i_patch][0, 0, :, :, :] = img_patch

        score = Net.predict(patches + [inputs, cams])
        cls_score.append(score)
    
    sio.savemat(args.save_path + 'hybnet_val_score.mat', 
                {'val_score': cls_score})


