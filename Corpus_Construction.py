# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:41:24 2020

@author: LG
"""

import numpy as np
import pickle
from matplotlib import pyplot as plt

np.random.seed(3)
#alpha from 6.2 of paper
alpha = np.array([0.1,0.1,0.1,0.1,0.1])

#beta from exponential distribution (not sure)

beta = np.zeros((100,5))
lam = 0.2
for i in range(5):
    unif = np.random.uniform(0,1,100)
    exp = -np.log(1-unif)/lam
    beta[0:100,i] = exp / np.sum(exp)

'''
beta = np.zeros((100,5))
beta[0:20,0] = np.array([0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,
                         0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]) 
beta[20:40,1] = np.array([0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,
                         0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]) 
beta[40:60,2] = np.array([0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,
                         0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]) 
beta[60:80,3] = np.array([0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,
                         0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]) 
beta[80:100,4] = np.array([0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,
                         0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]) 
'''
plt.subplot(511)
plt.bar(range(100),beta[:,0])
plt.subplot(512)
plt.bar(range(100),beta[:,1])
plt.subplot(513)
plt.bar(range(100),beta[:,2])
plt.subplot(514)
plt.bar(range(100),beta[:,3])
plt.subplot(515)
plt.bar(range(100),beta[:,4])
#plt.show();
plt.savefig("beta.png",dpi=300,bbox_inches="tight")

#top 10 words from each topic
top = np.zeros((10,5))
with open('true_beta_topten.pickle','wb') as f:
    for i in range(5):
        sort = np.argsort(-beta[0:100,i])
        top[0:10,i] = sort[0:10]
        pickle.dump(sort[0:10],f)
with open('true_beta.pickle','wb') as f:
    pickle.dump(beta,f)
#calculating TU value
TU = np.zeros(5)
unique, counts = np.unique(top,return_counts=True)
dic = dict(zip(unique,counts))
for i in range(5):
    for j in range(10):
        TU[i] += 0.1 / dic[top[j,i]]
print(TU)
#length of the document from poisson distribution
#setting document, save in corups.pickle

corpus = []
for i in range(10000):
    #w = [0 for i in range(100)]
    theta = np.random.dirichlet(alpha)
    length = np.random.poisson(lam=100000)
    x = np.dot(beta,theta)
    w = np.random.multinomial(length,x)
    w = w/length
    corpus.append(w)
with open('corpus.pickle','wb') as f:
    pickle.dump(corpus, f, pickle.HIGHEST_PROTOCOL)
    
corpus = []
for i in range(10000):
    #w = [0 for i in range(100)]
    theta = np.random.dirichlet(alpha)
    length = np.random.poisson(lam=100000)
    x = np.dot(beta,theta)
    w = np.random.multinomial(length,x)
    w = w/length
    corpus.append(w)
with open('test_corpus.pickle','wb') as f:
    pickle.dump(corpus, f, pickle.HIGHEST_PROTOCOL)
    


