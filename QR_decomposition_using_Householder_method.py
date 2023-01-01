# -*- coding: utf-8 -*-
"""
                       Term Paper
Created on Fri Dec  2 21:22:40 2022

@author: Amit Ranjan (222EE3184)
"""

import numpy as np

#QR-decomposition of rectangular matrix A using the decompostion tool
def decomposition(A):
    # Initialization of the orthogonal matrix Q and the upper triangular matrix R
    n, m = A.shape
    Q = np.eye(n)
    R = np.copy(A)

    rs, cs = np.tril_indices(n, -1, m)
    for (ro, cl) in zip(rs, cs):
        # If the subdiagonal element is nonzero, then compute the nonzero 
        # components of the rotation matrix
        if R[ro, cl] != 0:
            r = np.sqrt(R[cl, cl]**2 + R[ro, cl]**2)
            c, s = R[cl, cl]/r, -R[ro, cl]/r

            # The rotation matrix is highly discharged, so it makes no sense 
            # to calculate the total matrix product
            R[cl], R[ro] = R[cl]*c + R[ro]*(-s), R[cl]*s + R[ro]*c
            Q[:, cl], Q[:, ro] = Q[:, cl]*c + Q[:, ro]*(-s), Q[:, cl]*s + Q[:, ro]*c

    return Q[:, :m], R[:m]


def Householder(A):
    """
    QR-decomposition of a rectangular matrix A using the Householder method.
    """

    # Initialization of the orthogonal matrix Q and the upper triangular matrix R
    n, m = A.shape
    I = np.eye(n)
    R = np.copy(A)

    for i in range(m):
        v = np.copy(R[i:, i]).reshape((n-i, 1))
        v[0] = v[0] + np.sign(v[0]) * np.linalg.norm(v)
        v = v / np.linalg.norm(v)
        R[i:, i:] = R[i:, i:] - 2 * v @ v.T @ R[i:, i:]
        I[i:] = I[i:] - 2 * v @ v.T @ I[i:]

    return I[:m].T, R[:m]


# To check the solutions, we use the standard deviation of SME
def SME(A, b, x):
    return 1/max(b) * np.sqrt(1/len(b) * np.sum(abs(np.dot(A, x) - b) ** 2))



if __name__=='__main__':

    # Consider an example of a square matrix:
    
    A = np.array([[1,3,-1],[4,2,6],[-6,7,1]])
    b1 = np.array([1,-3,4])

    Q1, R1 = decomposition(A)
    x = np.linalg.solve(R1, Q1.T @ b1)
    print('Givens: ', SME(A, b1, x))

    Q1, R1 = Householder(A)
    x = np.linalg.solve(R1, Q1.T @ b1)
    print('Householder: ', SME(A, b1, x))
    print('Q matrix is :\n',Q1)
    print('R matrix is : \n',R1)

    # Now consider an example of Rectangulr matrix:
    # A = np.array([[1,2,-5],[4,-2,]])
    # b2 = np.array([1,3,-1])

    # Q2, R2 = decomposition(A)
    # x = np.linalg.solve(R2, Q2.T @ b2)
    # print('Rectangular Givens: ', SME(A, b2, x))

    # Q2, R2 = Householder(A)
    # x = np.linalg.solve(R2, Q2.T @ b2)
    # print('Rectangular Householder: ', SME(A, b2, x))