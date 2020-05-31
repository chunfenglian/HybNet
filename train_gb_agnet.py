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
from network import AttGatedNet
from feed_data import Stage2GlobalLoader
import argparse

parser = argparse.ArgumentParser(description='Train the attention-gated sub-network in the global branch.')
parser.add_argument('--with_bn', type=bool, default=True)
parser.add_argument('--with_drop', type=bool, default=True)
parser.add_argument('--drop_prob', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_outputs', type=int, default=1)

parser.add_argument('--img_path', default='/shenlab/lab_stor4/cflian/norm_Crop_AD1_ADNC/')
parser.add_argument('--cam_path', default='/shenlab/lab_stor4/cflian/cam_ad1_adnc/')
parser.add_argument('--img_size', default=[144, 184, 152])
parser.add_argument('--save_path', default='./saved_model/')
parser.add_argument('--model_name', default='stage2_attgatednet')
parser.add_argument('--gpu', type=int, default=0)

NUM_GB_FEATURES = 64

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

    Net = AttGatedNet(image_size=args.img_size, num_chns=1, 
                      num_outputs=args.num_outputs, 
                      feature_depth=[16, 16, 32, 32, 64, 64, 128, 128], 
                      gb_chns=NUM_GB_FEATURES,
                      with_bn=args.with_bn, with_dropout=args.with_drop, 
                      drop_prob=args.drop_prob).agnet()
    Net.get_layer('backbone').trainable = False
    Net.get_layer('backbone').load_weights(args.save_path + 'stage1_backbone.best.h5')
    Net.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    Net.summary()

    generator = Stage2GlobalLoader(image_path=args.img_path,
                                  cam_path=args.cam_path, 
                                  batch_size=args.batch_size,
                                  num_input_chns=1,
                                  num_outputs=args.num_outputs,
                                  image_size=args.img_size,
                                  scale_range=[0.95, 1.05],
                                  flip_axis=0,
                                  shuffle_flag=True)
    train_flow = generator.data_flow(TRN_SUBJ_LIST, TRN_SUBJ_LBLS, 
                                     flip_flag=True, noise_flag=True)
    validate_flow = generator.data_flow(VAL_SUBJ_LIST, VAL_SUBJ_LBLS, 
                                        flip_flag=False, noise_flag=False)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
                      filepath=args.save_path + args.model_name + '.best.h5',
                      save_weights_only=True,
                      monitor='val_accuracy', mode='max', save_best_only=True)
    logger = keras.callbacks.CSVLogger(args.save_path + args.model_name + '.log', 
                                      separator=",", append=False)
    
    # TRAIN max_queue_size=10
    Net.fit_generator(generator=train_flow,
                      steps_per_epoch=trn_steps,
                      epochs=args.num_epochs,
                      validation_data=validate_flow,
                      validation_steps=val_steps,
                      callbacks=[checkpoint, logger])
