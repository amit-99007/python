# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 19:53:28 2022

@author: Amit Ranjan (222EE3184)
"""

import numpy as np

def row_echelon(A):
    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    n, m = A.shape
    if n == 0 or m == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we exchange rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row  
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

A = np.array([[0,1,3,4],[0,4,5,6],[0,8,9,10],[5,3,4,2]], dtype='float')

row_echelon(A)
print('row_echelon form is : \n',A)