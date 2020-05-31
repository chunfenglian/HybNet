# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:22:43 2018

@author: chlian
"""

import keras
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio
from keras.models import Model
from network import CAM
from feed_data import SequentialLoader
from sklearn.cross_validation import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Train the backbone FCN for CAM generation.')
parser.add_argument('--with_bn', type=bool, default=True)
parser.add_argument('--with_drop', type=bool, default=True)
parser.add_argument('--drop_prob', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_outputs', type=int, default=1)

parser.add_argument('--img_path', default='/shenlab/lab_stor4/cflian/norm_Crop_AD1_ADNC/')
parser.add_argument('--img_size', default=[144, 184, 152])
parser.add_argument('--resume', default='./saved_model/stage1_pretrain.best.h5')
parser.add_argument('--save_path', default='./cams/')
parser.add_argument('--gpu', type=int, default=0)

DATA_PARTITION = sio.loadmat('files/'+'data_partition.mat')
TRN_SUBJ_LIST = DATA_PARTITION['trn_list'][0].tolist()
TRN_SUBJ_LBLS = DATA_PARTITION['trn_lbls'][0].tolist()
VAL_SUBJ_LIST = DATA_PARTITION['val_list'][0].tolist()
VAL_SUBJ_LBLS = DATA_PARTITION['val_lbls'][0].tolist()

TRN_FLAG = False

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
      os.makedirs(args.save_path)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    Net = CAM(image_size=args.img_size, num_chns=1, 
              num_outputs=args.num_outputs, 
              feature_depth=[16, 16, 32, 32, 64, 64, 128, 128], 
              with_bn=args.with_bn, with_dropout=args.with_drop, 
              drop_prob=args.drop_prob,
              trn_flag=TRN_FLAG).forward()
    Net.load_weights(args.resume)

    weights = Net.get_layer(name='subject_outputs').get_weights()[0]
    
    ad_score = []
    mean_map, counter = 0.0, 0    
    for i in range(len(TRN_SUBJ_LIST)):
        
        i_subj = TRN_SUBJ_LIST[i]
        print args.img_path + 'Crop_Img_{0}.nii.gz'.format(i_subj)
        
        img_dir = args.img_path + 'Crop_Img_{0}.nii.gz'
        img = np.array(sitk.GetArrayFromImage(sitk.ReadImage(img_dir.format(i_subj))))
        
        input = np.zeros(shape=[1,] *2 + args.img_size, dtype='float32')
        input[0, 0, :, :, :] = img
        [score, maps] = Net.predict(input)
        ad_score.append(score)
        
        cam = np.zeros(shape=np.shape(maps[0, 0, :, :, :]), dtype='float32')
        for i_chn in range(np.shape(weights)[0]):
            cam += weights[i_chn, 0] * maps[0, i_chn, :, :, :]
        
        cam = nd.interpolation.zoom(cam, 8, order=1)
        cam_max, cam_min = np.max(cam), np.min(cam)
        cam = (cam - cam_min) / (cam_max - cam_min)

        if TRN_SUBJ_LBLS[i] == 1:
            mean_map += cam
            counter += 1  

        co_resolution = np.array([1, 1, 1])
        HM = sitk.GetImageFromArray(cam, isVector=False)
        HM.SetSpacing(co_resolution)
        sitk.WriteImage(HM, args.save_path + 'CAM_AD1_ADNC_Subj{0}.nii.gz'.format(i_subj))

    mean_map = mean_map / counter
    HM = sitk.GetImageFromArray(cam, isVector=False)
    HM.SetSpacing(co_resolution)
    sitk.WriteImage(HM, args.save_path + 'Mean_CAM.nii.gz')

    tvalue = 0.3
    tm = np.zeros(mean_map.shape)
    tm[mean_map>tvalue] = 1
    tm[mean_map<=tvalue] = 0

    idxs = np.zeros((3, len(np.nonzero(tm)[0])), dtype='int8')
    idxs1 = np.nonzero(tm)[0]
    idxs2 = np.nonzero(tm)[1]
    idxs3 = np.nonzero(tm)[2]
    c_value = np.zeros((1, len(np.where(tm==1)[0])), dtype='float32')
    for i in range(0, len(np.where(tm==1)[0])):
        c_value[0, i] = mean_map[idxs1[i], idxs2[i], idxs3[i]]
    sio.savemat(args.save_path + 'initial_patch_locations.mat',
            {'location1': idxs1, 'location2': idxs2, 'location3': idxs3, 'c_values': c_value})

    sio.savemat('./saved_model/' + 'train_score.mat', 
                {'train_score': ad_score})