# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:39:19 2020

@author: LG
"""
from wae import WAE
from wae_modified import WAE_modified

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt


encoder = keras.models.load_model('encoder.h5')
decoder = keras.models.load_model('decoder.h5')
#wae = WAE(encoder,decoder)
wae = WAE(encoder,decoder)

encoder_modified = keras.models.load_model('encoder_modified.h5')
decoder_modified = keras.models.load_model('decoder_modified.h5')
wae_modified = WAE_modified(encoder_modified,decoder_modified)

with open('corpus.pickle','rb') as f:
    corpus = pickle.load(f)

d = 21
#print(corpus[0])
theta = encoder(tf.constant(corpus[d:d+1]))
reconstruct = decoder(theta)
theta_modified = encoder_modified(tf.constant(corpus[d:d+1]))
reconstruct_modified = decoder_modified(theta_modified)
reconstruct_modified = tf.divide(reconstruct_modified,tf.reduce_sum(reconstruct_modified,axis=1))
#print(reconstruct)

plt.bar(range(100),corpus[d])
plt.show()
plt.bar(range(100),reconstruct[0,:])
plt.show()
plt.bar(range(100),reconstruct_modified[0,:])
plt.show()

print(wae.reconstructionLoss(tf.constant([corpus[d]]),reconstruct))
print(wae.reconstructionLoss(tf.constant([corpus[d]]),reconstruct_modified))


beta = wae_modified.decoder.layers[1].weights

plt.subplot(511)
plt.bar(range(100),beta[0][0,:])
plt.subplot(512)
plt.bar(range(100),beta[0][1,:])
plt.subplot(513)
plt.bar(range(100),beta[0][2,:])
plt.subplot(514)
plt.bar(range(100),beta[0][3,:])
plt.subplot(515)
plt.bar(range(100),beta[0][4,:])
#plt.show();
plt.savefig("model_beta.png",dpi=300,bbox_inches="tight")






