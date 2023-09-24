"""
Utility module for optimization repo
"""

import numpy as np

def tridiagonal_solve(a, b, c, d):
    """
    Reference: wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    All four inputs are length-n 1D numpy arrays.
    Elements/entries a[n-1] and c[0] are not used (undefined).
    """
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

def tridiag_solve_specialcase(b: np.ndarray, d: np.ndarray): 
    """
    Specialization with a = c = -0.5
    Side-effect: Destroying inputs along the way.
    """
    n = len(d)
    for i in range(n-1):
        frac = -0.5/b[i]
        d[i+1] -= d[i] * frac
        b[i+1] -=  -0.5 * frac
    for i in range(n-1,0,-1): # reversed
        d[i-1] -= (d[i] * -0.5) / b[i]
    return d / b