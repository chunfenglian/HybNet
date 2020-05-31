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
parser.add_argument('--save_path', default='./saved_model/')
parser.add_argument('--model_name', default='stage1_backbone')
parser.add_argument('--gpu', type=int, default=0)

SUBJECT_IDXS = range(1, 429)
LABEL_NAME = 'labels_ADNI1'
LABELS = sio.loadmat('files/'+LABEL_NAME+'.mat')
LABELS = LABELS[LABEL_NAME]
LABELS[np.where(LABELS==-1)[0]] = 0

NEG_SUBJ_IDXS = [SUBJECT_IDXS[i] for i in np.where(LABELS==0)[0]]
POS_SUBJ_IDXS = [SUBJECT_IDXS[i] for i in np.where(LABELS==1)[0]]
NEG_LABELS = [LABELS[i][0] for i in np.where(LABELS==0)[0]]
POS_LABELS = [LABELS[i][0] for i in np.where(LABELS==1)[0]]

TRN_NEG_IDXS, VAL_NEG_IDXS, TRN_NEG_LBLS, VAL_NEG_LBLS = \
    train_test_split(NEG_SUBJ_IDXS, NEG_LABELS, test_size=0.1)
TRN_POS_IDXS, VAL_POS_IDXS, TRN_POS_LBLS, VAL_POS_LBLS = \
    train_test_split(POS_SUBJ_IDXS, POS_LABELS, test_size=0.1)

TRN_SUBJ_LIST = TRN_NEG_IDXS + TRN_POS_IDXS
TRN_SUBJ_LBLS = TRN_NEG_LBLS + TRN_POS_LBLS
VAL_SUBJ_LIST = VAL_NEG_IDXS + VAL_POS_IDXS
VAL_SUBJ_LBLS = VAL_NEG_LBLS + VAL_POS_LBLS

sio.savemat('files/'+'data_partition.mat', 
            {'trn_list': TRN_SUBJ_LIST, 
            'trn_lbls': TRN_SUBJ_LBLS,
            'val_list': VAL_SUBJ_LIST,
            'val_lbls': VAL_SUBJ_LBLS})

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
      os.makedirs(args.save_path)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    trn_steps = int(np.round(len(TRN_SUBJ_LIST) / args.batch_size))
    val_steps = int(np.round(len(VAL_SUBJ_LIST) / args.batch_size))

    print('Trianing {0} iterations in each epoch'.format(trn_steps))

    Net = CAM(image_size=args.img_size, num_chns=1, 
              num_outputs=args.num_outputs, 
              feature_depth=[16, 16, 32, 32, 64, 64, 128, 128], 
              with_bn=args.with_bn, with_dropout=args.with_drop, 
              drop_prob=args.drop_prob).forward()
    #Net.load_weights(params['model_path']+'stage1_initialization.best.h5')

    generator = SequentialLoader(image_path=args.img_path, 
                                 batch_size=args.batch_size,
                                 num_input_chns=1,
                                 num_outputs=args.num_outputs,
                                 image_size=args.img_size,
                                 scale_range=[0.95, 1.05],
                                 flip_axis=0,
                                 shuffle_flag=True)
    train_flow = generator.data_flow(TRN_SUBJ_LIST, TRN_SUBJ_LBLS, 
                                     scale_flag=False, flip_flag= True)
    validate_flow = generator.data_flow(VAL_SUBJ_LIST, VAL_SUBJ_LBLS, 
                                        scale_flag=False, flip_flag= False)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
                      filepath=args.save_path + args.model_name + '.best.h5',
                      save_weights_only=True,
                      save_best_only=True, mode='auto')
    logger = keras.callbacks.CSVLogger(args.save_path + args.model_name + '.log', 
                                      separator=",", append=False)
    
    # TRAIN max_queue_size=10
    Net.fit_generator(generator=train_flow,
                      steps_per_epoch=trn_steps,
                      epochs=args.num_epochs,
                      validation_data=validate_flow,
                      validation_steps=val_steps,
                      callbacks=[checkpoint, logger])