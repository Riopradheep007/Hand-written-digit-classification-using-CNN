#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:32:52 2020

@author: kpr
"""

import cv2
import tensorflow as tf


def prepare(filepath):
    IMG_SIZE=50
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model=tf.keras.models.load_model('weight1.h5')

prediction=model.predict([prepare('/home/kpr/Music/deep learning/digits classification/pradheep_model/database/8/train_38_00008.png')])#path of the image you need to classify
print(prediction)
