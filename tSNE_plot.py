# -*- coding: utf-8 -*-

# code by Luuk Derksen
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

from __future__ import print_function
import time

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import tensorflow as tf
from tensorflow import keras

## Download MNIST mat file from GitHub

from wae import WAE
latent = 10
n_elements = [100,50,latent]
alpha = [0.1,0.1,0.1,0.1,0.1]

tf.keras.backend.set_floatx('float64')

encoder = keras.models.load_model('encoder.h5')
decoder = keras.models.load_model('decoder.h5')
wae = WAE(encoder,decoder)

## N is the number of dots in t-SNE

def tSNE_plot(wae, N=3000):
    with open('corpus.pickle','rb') as f:
        corpus = pickle.load(f)
        corpus_divided = tf.convert_to_tensor(corpus[0:N])
    ## loading data
    X1 = wae.encoder(corpus_divided)
    X2 = tf.constant(np.random.dirichlet(wae.alpha,size=N))
    X = tf.concat([X1,X2],0)
    y = ['model'] * N + ['real'] * N

    ## Convert matrix/vector to Pandas DataFrame
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))
    
    ## For Reproducibility  of the results
    rndperm = np.random.permutation(df.shape[0])
        
    ## t-SNE 
    # Sampled data (N=10000)
    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values
    
    # t-SNE
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    colors = ["#108226", "#e74c3c"]
    
    # plot
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette(colors),
        data=df_subset,
        legend="full",
        alpha=1
    )
    plt.show()

np.random.seed(42)

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

wae = WAE(encoder,decoder,alpha=[0.1]*latent)
wae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01,beta_1=0.99,beta_2=0.999))
for i in range(10):
    wae.fit(tf.constant(corpus[i*1000:(i+1)*1000]), epochs=10, batch_size=100)
    print("epoch", i+1)
    tSNE_plot(wae)