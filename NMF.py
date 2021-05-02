# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:45:17 2020

@author: LG
"""
import pickle
from matplotlib import pyplot as plt
#from nmf import NMF
import numpy as np
import sklearn
from sklearn.decomposition import NMF
from tensorflow import keras


with open('corpus.pickle','rb') as f:
    corpus = pickle.load(f)


corpus = np.array(corpus)
model = NMF(n_components=5, init=None, solver='cd', beta_loss='frobenius',tol=0.0001, 
            max_iter=1000, random_state=None, alpha=0.0, l1_ratio=0.0, verbose=0, shuffle=False,)
W = model.fit_transform(corpus.transpose()).transpose()
#print(W)
for i in range(5):
    plt.bar(range(100),W[i,:])
    plt.show()

decoder = keras.models.load_model('decoder.h5')    
beta = decoder.layers[1].weights
with open('true_beta_topten.pickle','rb') as f:
    model = []
    real = []
    NMF = []
    for i in range(5):
        model.append(np.argsort(beta[0][i,:])[::-1][0:10])
        real.append(pickle.load(f))
        NMF.append(np.argsort(W[i,:])[::-1][0:10])
    print("\n model beta")        
    print(model)
    print("\nreal beta")
    print(real) 
    print("\nNMF beta")
    print(NMF)










