# -*- coding: utf-8 -*-
"""
                    QR Decomposition
Created on Fri Nov  28 17:22:40 2022

@author: Amit Ranjan (222EE3184)
"""

import numpy as np

# Define A matrix
 
A = np.array([[1,3,-1],[4,2,6],[-6,7,1]])

# Slice out the columns of A for processing
a = A[:,0]
b = A[:,1]
c = A[:,2]


# Carry out Gram-Schmidt process
 
X = a
q_1 = X/np.linalg.norm(a)
 
 
Y = b - np.dot(b,q_1)*q_1
q_2 = Y/np.linalg.norm(Y)
 

Z = c - np.dot(c,q_1)*q_1 - np.dot(c,q_2)*q_2
q_3 =Z/ np.linalg.norm(Z)
 


#  Assemble the matrix Q.

Q = np.vstack([q_1,q_2,q_3])

print("The Obtained Q is\n" ,Q,'\n')

#  Checking that if Q is orthogonal.

print("QTQ")
print(np.round(Q.transpose()@Q),'\n')

#  Computation of R.

R = Q.transpose()@A
print("R \n",np.round(R,4),'\n')

#  Checking A from Q and R.
print("A \n", A)
print("QR \n",np.round(Q@R))
