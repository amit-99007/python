# -*- coding: utf-8 -*-
"""
             Question :6
Created on Fri Dec  2 21:22:40 2022

@author: Amit Ranjan (222EE3184)
"""

import numpy as np
import matplotlib.pyplot as plt

#Function to define noise 

def noise(m,v,s,d):                               #define noise
    sig = np.sqrt(v)                              #calculation of sigma
    if d <= 1:                                    #checking if else condition
        return m + sig*np.random.randn(s[0])
    else:
        return m + sig*np.random.randn(s[0],s[1])
    
#Function to perform DTFT 

def DTFT(S):                #define discrete time fourier transform
    X = len(S)                                      #length of X
    print(X)
    n = np.arange(X)                                #arange X in n
    print(n)
    k = n.reshape(X,1)
    w = (2*(np.pi)*k)/X
    e = np.exp(-1j*w*n)
    D = np.dot(S,e)                                #dot product of S and e
    return D
#Noise Parameters

Wm =0             #Mean = 0
Wv = 2                #Variance = 1
Nreal = 3000     #Number of realizations
T = 5                #Number of samples
s = (Nreal,T)                  #s = (length of Nreal,length of T)
t = np.arange(0,T,1)           # t = 0 to T
N = noise(Wm,Wv,s,2)           #Feeding all parameters to noise function

#%%
#Ensamble parameters of noise N

Em = np.sum(N,axis = 0)/Nreal             #Ensemble mean, Em
R = np.dot(np.transpose(N),N)/Nreal       #E(N,N) co-rrelational matrix
Evr= np.diag(R)                          #Ensemble variance
r = R[0,:]                                #Auto Co-rrelational sequence of r
PSD = DTFT(r)                             #Power Spectral Density of r
M_PSD = abs(PSD)                          #Absolute magnitude of PSD
Anfreq=np.fft.fftfreq(T)*(2*np.pi)        #Angular frequency

#%%
#Required plot

plt.figure(1)
plt.show()
plt.xlabel('Angular Freq(W)')
plt.ylabel('|PSD|')
plt.stem(Anfreq,M_PSD,'b',markerfmt = "*",basefmt = '-r')


