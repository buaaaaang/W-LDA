# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:52:54 2020

@author: LG
"""

from wae import WAE
from wae_modified import WAE_modified

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import itertools

tf.keras.backend.set_floatx('float64')

def newModel(n_elements,noise,lamb):
    encoder_input = keras.Input(shape=(n_elements[0]))
    encoder_layer = keras.layers.Dense(n_elements[1],activation="relu")(encoder_input)
    for i in range(2,len(n_elements)-1):  
        encoder_layer = keras.layers.Dense(n_elements[i],activation="relu")(encoder_layer)
    encoder_output = keras.layers.Dense(n_elements[-1],activation="softmax")(encoder_layer)
    encoder = keras.Model(encoder_input,encoder_output,name="encoder")
            
    decoder_input = keras.Input(shape=(n_elements[-1]))
    #decoder_output = keras.layers.Dense(n_elements[0],kernel_initializer='zeros',activation="softmax")(decoder_input)
    decoder_output = keras.layers.Dense(n_elements[0],activation="softmax")(decoder_input)
    decoder = keras.Model(decoder_input,decoder_output,name="decoder")

    with open('corpus.pickle','rb') as f:
        corpus = pickle.load(f)

    wae = WAE(encoder,decoder,noise,lamb)
    wae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01,beta_1=0.99,beta_2=0.999))
    wae.fit(tf.constant(corpus), epochs=100, batch_size=100)
    beta = wae.decoder.layers[1].weights[0]
    
    with open('true_beta.pickle','rb') as f:
        real = tf.constant(pickle.load(f))
    
    P = itertools.permutations(range(5))
    loss = None
    for p in P:
        l = 0.
        for i in range(5):
            l = l + tf.reduce_sum(tf.abs((tf.nn.softmax(beta[p[i],:])-real[:,i]))).numpy()
            #l = l + tf.nn.l2_loss(beta[p[i],:]-real[:,i]).numpy()
        if loss==None:
            loss = l
        else:
            if loss > l:
                loss = l
    return loss

def newModel2(n_elements,noise,lamb):
    encoder_input = keras.Input(shape=(n_elements[0]))
    encoder_layer = keras.layers.Dense(n_elements[1],activation="relu")(encoder_input)
    for i in range(2,len(n_elements)-1):  
        encoder_layer = keras.layers.Dense(n_elements[i],activation="relu")(encoder_layer)
    encoder_output = keras.layers.Dense(n_elements[-1],activation="softmax")(encoder_layer)
    encoder = keras.Model(encoder_input,encoder_output,name="encoder")
            
    decoder_input = keras.Input(shape=(n_elements[-1]))
    decoder_output = keras.layers.Dense(n_elements[0],activation="relu",kernel_constraint=keras.constraints.NonNeg(),
                                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.1))(decoder_input)
    decoder = keras.Model(decoder_input,decoder_output,name="decoder")

    with open('corpus.pickle','rb') as f:
        corpus = pickle.load(f)

    wae = WAE_modified(encoder,decoder,noise,lamb)
    wae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01,beta_1=0.99,beta_2=0.999))
    wae.fit(tf.constant(corpus), epochs=100, batch_size=100)
    beta = wae.decoder.layers[1].weights[0]
    
    with open('true_beta.pickle','rb') as f:
        real = tf.constant(pickle.load(f))
    
    P = itertools.permutations(range(5))
    loss = None
    for p in P:
        l = 0.
        for i in range(5):
            l = l + tf.reduce_sum(tf.abs(beta[p[i],:]/tf.reduce_sum(beta[p[i],:])-real[:,i])).numpy()
            
        if loss==None:
            loss = l
        else:
            if loss > l:
                loss = l
    return loss


import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


loss= 10
for i in range(100):
    h = int(np.random.uniform(1,3))
    n_elements = [100]
    for i in range(h):
        u = n_elements[-1]
        n_elements.append(int(np.random.uniform(5,u+1)))
    n_elements.append(5)
    #noise = np.random.uniform(0,0.5)
    noise = 0.
    lamb = 1.
    l = 0
    n=2
    with HiddenPrints():
        for i in range(n):
            l += newModel2(n_elements,noise,lamb)
    l = l/n        
    print(n_elements,':',l)
    if l < loss:
        loss = l
        print()

'''
loss = []
n = 5
noise = 0.
lamb = 1.
l = 0
with HiddenPrints():
    for i in range(n):
        l += newModel([100,100,50,5],noise,lamb)
    loss.append(l/n)
print(l/n)

noise = []
loss = []
for i in range(16):
    n = i/20
    noise.append(n)
    lamb = 1
    l = 0.
    with HiddenPrints():
        for i in range(5):
            l += newModel([100,100,50,5],n,lamb)
    loss.append(l/5)
    print(loss[-1])
plt.plot(noise,loss)
plt.show()
'''
    


    
        
    



            
            
                
    








