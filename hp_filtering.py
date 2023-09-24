"""
Hodrick-Prescott filtering

TODO:
- Write tests for tridiagonal_solve, the code is most likely buggy.
- Implement HP-filtering without realizing the matrix involved.
"""

import numpy as np

def build_D_matrix(n: int, dtype='float32'):
    """
    Explicit realization of the (n-2) x n second-order diff matrix D.
    """
    assert (n >= 3), 'Dimension n too small'
    D = np.zeros([n-2,n], dtype=dtype)
    row = np.r_[1.,-2.,1.,np.zeros(n-3, dtype=dtype)]
    for i in range(n-2):
        D[i,:] = np.roll(row, i)
    
    return D

def build_DtopD_matrix(n: int, dtype='float32'):
    """
    Explicit realization of n x n matrix D.transpose().dot(D).
    """
    assert (n >= 5), 'Dimension n too small'
    DtopD = np.zeros([n,n], dtype=dtype)
    # handle first rwo rows and last two rows
    DtopD[0,0:3] = (1.,-2.,1.)
    DtopD[1,0:4] = (-2.,5.,-4.,1.)
    DtopD[n-2,-4:] = (1.,-4.,5.,-2.) # reversed order
    DtopD[n-1,-3:] = (1.,-2.,1.)
    # all other rows
    row = np.r_[1.,-4.,6.,-4.,1.,np.zeros(n-5, dtype=dtype)]
    for i in range(2,n-2):
        DtopD[i,:] = np.roll(row, i-2)
    
    return DtopD

def hp_filter_explicit(y: np.ndarray, lamdba_=1000.):
    """
    Hodrick-Prescott filtering with explicit realization of matrix.
    """
    n = len(y)    
    A = eye(n) + (2.*lambda_)*build_DtopD_matrix(n)
    x_hp = np.linalg.solve(A,y)
    return x_hp

def tridiagonal_solve(a, b, c, d):
    # wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    # all four inputs are length-n 1D numpy arrays
    # a[n-1] and c[0] are not used (undefined)
    n = len(d)
    bb = b.copy() # avoid destroying input
    dd = d.copy()
    for i in range(n-1):
        frac = a[i]/bb[i]
        dd[i+1] -= dd[i] * frac
        bb[i+1] -=  c[i] * frac
    for i in range(n-2,-1,-1): # reversed
        dd[i] -= (dd[i+1] * c[i]) / bb[i+1]
    return dd / bb

