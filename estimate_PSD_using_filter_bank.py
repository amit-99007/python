# -*- coding: utf-8 -*-
"""
              Question : 7
Created on Fri Dec  2 21:22:40 2022

@author: Amit Ranjan (222EE3184)
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

Ns =5                             #number of samples
Nr =3000                          #number of realizations
mean =0                           #mean =0
var =0.2                          #variance =0.2

#Function for Guass White Noise

def wgn(n,Nr,m,Vr):
    p = np.random.randn(Nr,n)    #Taking p random variables
    v = m + Vr*p                 #V(n)
    temp = plotPSD(v,m,Vr,n,Nr)   #Design function to plot graphs
    if temp == 0:                 #Checking condition
        return v
    else:
        print("error")
        return 0;


def plotPSD(Xn,m,Vr,Ns,Nr):                  #Function designed to plot graphs
    cor_vn = np.dot(np.transpose(Xn),Xn)/Nr  #Ensemble covariance of V(n)
    Evarx = np.diag(cor_vn)                  #Ensemble variance of V(n)
    AuCorvx = cor_vn[0,:]                    #Auto co-rrelation sequence
    Rx = np.fft.fft(AuCorvx)                 #fft of V(n)
    freq1 = np.fft.fftfreq(Ns)               #1st frequency
    plt.figure()                             #plot figure
    plt.stem(2*np.pi*freq1,abs(Rx),'r')      #stem plot
    plt.ylabel('|PSD|')                      #y-axis label
    plt.xlabel('Angular Frequency')          #x-axis label
    plt.title("mean = %d" % m + "and variance = %f" % Vr)
    plt.grid();                              #introducing grid in figure
    print(max(Evarx))                        #print maximum Ensemble variance
    return 0


WN = wgn(Ns,Nr,mean,var)                     #Guass white noise variables

#%% for different signal frequency
#For bandpass 2 selected [f1,f2] and fs(total sample/step)of the bandpass filter
# Output='sos' -> second order sections

#(1)
S1 = signal.butter(2,[10,20],'bandpass',fs = 1000,output='sos')
filtered1 = signal.sosfilt(S1,WN)
temp = plotPSD(filtered1,mean,var,Ns,Nr)

#(2)
S2 = signal.butter(2,[20,30],'bandpass',fs = 1000,output='sos')
filtered2 = signal.sosfilt(S2,WN)
temp = plotPSD(filtered2,mean,var,Ns,Nr)

#(3)
S3 = signal.butter(2,[30,40],'bandpass',fs = 1000,output='sos')
filtered3 = signal.sosfilt(S3,WN)
temp = plotPSD(filtered3,mean,var,Ns,Nr)

#(4)
S4 = signal.butter(2,[40,50],'bandpass',fs = 1000,output='sos')
filtered4 = signal.sosfilt(S4,WN)
temp = plotPSD(filtered4,mean,var,Ns,Nr)

#(5)
S5 = signal.butter(2,[50,60],'bandpass',fs = 1000,output='sos')
filtered5 = signal.sosfilt(S5,WN)
temp = plotPSD(filtered5,mean,var,Ns,Nr)

