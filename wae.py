# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:32:26 2020

@author: LG
"""
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import itertools

tf.keras.backend.set_floatx('float64')

n_elements = [100,100,50,5]
alpha = [0.1,0.1,0.1,0.1,0.1]

class WAE(keras.Model):
    def __init__(self, encoder, decoder, noise=0.2, lamb=1.,alpha=[0.1]*5, **kwargs):
        super(WAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.noise = noise
        self.lamb = lamb
        self.alpha=alpha

    def reconstructionLoss(self,w,w_hat):
        #getting reconstruction loss (cross-entropy loss)
        batch_size = w.shape[0]
        eps = 10**-12
        w_hat = tf.clip_by_value(w_hat,eps,1)
        log_w_hat = tf.math.log(w_hat)
        c = tf.multiply(w,log_w_hat)
        c = tf.reduce_sum(c)
        return -1. * c/batch_size
        #return tf.reduce_sum(tf.math.square(w-w_hat))
    
    def diffusion_kernel(self,tensor):
        return tf.math.exp(-1*tf.math.square(tf.math.acos(tensor)))
    
    def mmdLoss(self,false_theta,true_theta):
        #getting MMD loss
        batch_size = false_theta.shape[0]
        assert batch_size==100, "wrong batch size"
        eps = 10**-12
        f = tf.math.sqrt(tf.clip_by_value(false_theta,eps,1))
        t = tf.math.sqrt(tf.clip_by_value(true_theta,eps,1))
        ft = tf.clip_by_value(tf.matmul(f,t,transpose_b=True),0,1-eps)
        ff = tf.clip_by_value(tf.matmul(f,f,transpose_b=True),0,1-eps)
        tt = tf.clip_by_value(tf.matmul(t,t,transpose_b=True),0,1-eps)
        ft = self.diffusion_kernel(ft)
        ff = self.diffusion_kernel(ff)
        tt = self.diffusion_kernel(tt)
        ft = 2*tf.reduce_sum(ft)/(batch_size**2)
        ff = tf.reduce_sum(ff)/(batch_size*(batch_size-1))
        tt = tf.reduce_sum(tt)/(batch_size*(batch_size-1))
        return tt + ff - ft

    def train_step(self, w):
        with tf.GradientTape() as tape:
            theta = self.encoder(w)
            theta_noise = tf.constant(np.random.dirichlet(self.alpha,size=w.shape[0]))
            theta_plus = (1-self.noise)*theta + self.noise*theta_noise
            w_hat = self.decoder(theta_plus)
            true_theta = tf.constant(np.random.dirichlet(self.alpha,size=w.shape[0]))
            loss_reconstruction = self.reconstructionLoss(w,w_hat) 
            loss_mmd = self.mmdLoss(theta,true_theta)
            loss = loss_reconstruction + loss_mmd*self.lamb
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": loss,
            "reconstruction loss": loss_reconstruction,
            "mmd loss": loss_mmd,
        }

if __name__=='__main__':
    np.random.seed(3)
    encoder_input = keras.Input(shape=(n_elements[0]))
    encoder_layer = keras.layers.Dense(n_elements[1],activation="relu")(encoder_input)
    for i in range(2,len(n_elements)-1):  
        encoder_layer = keras.layers.Dense(n_elements[i],activation="relu")(encoder_layer)
    encoder_output = keras.layers.Dense(n_elements[-1],activation="softmax")(encoder_layer)
    encoder = keras.Model(encoder_input,encoder_output,name="encoder")
            
    decoder_input = keras.Input(shape=(n_elements[-1]))
    decoder_output = keras.layers.Dense(n_elements[0],activation="softmax")(decoder_input)
    decoder = keras.Model(decoder_input,decoder_output,name="decoder")

    with open('corpus.pickle','rb') as f:
        corpus = pickle.load(f)

    wae = WAE(encoder,decoder)
    wae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001,beta_1=0.99,beta_2=0.999))

    hist = wae.fit(tf.constant(corpus), epochs=100, batch_size=100)
    
    #top-ten words of topics
    beta = wae.decoder.layers[1].weights[0]
    '''
    with open('true_beta_topten.pickle','rb') as f:
        model = []
        real = []
        for i in range(5):
            model.append(np.argsort(beta[i,:])[::-1][0:10])
            real.append(pickle.load(f))
        print("\n model beta")        
        print(model)
        print("\nreal beta")
        print(real)   
    '''
    #beta matching
    with open('true_beta.pickle','rb') as f:
        real = tf.constant(pickle.load(f))
    P = itertools.permutations(range(5))
    loss = None
    for p in P:
        l = 0.
        for i in range(5):
            l = l + tf.reduce_sum(tf.abs(tf.nn.softmax(beta[p[i],:])-real[:,i])).numpy()
            #l = l + tf.nn.l2_loss(beta[p[i],:]-real[:,i]).numpy()

        if loss==None:
            loss = l
            p_true = p
        else:
            if loss > l:
                loss = l
                p_true = p
    print('l1 distance', loss)
    for i in range(5):
        print('topic',i+1)
        print('real:  ', np.argsort(real[:,i].numpy())[::-1][0:10])
        print('model: ', np.argsort(beta[p_true[i],:].numpy())[::-1][0:10])
                    

    total_loss = hist.history["loss"]
    reconstruction_loss = hist.history["reconstruction loss"]
    mmd_loss = hist.history["mmd loss"]
    plt.plot(total_loss)
    plt.show()
    plt.plot(reconstruction_loss)
    plt.show()
    plt.plot(mmd_loss)    
    plt.ylim(bottom=0)
    plt.show()   

    encoder.save('encoder.h5')
    decoder.save('decoder.h5')
    

