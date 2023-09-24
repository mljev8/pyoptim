#distutils: language=c
#cython: language_level=3
#cython: nonecheck=False, boundscheck=False, wraparound=True, cdivision=True

# run-time imports
import numpy as np

def tridiag_solve_specialcase(double[::1] b, double[::1] d):
    """
    Specialization with a = c = -0.5
    Side-effect: Destroying inputs along the way.
    """
    cdef size_t n = len(d)
    cdef size_t i 
    cdef float frac

    for i in range(n-1):
        frac = -0.5/b[i]
        d[i+1] -= d[i] * frac
        b[i+1] -=  -0.5 * frac
    for i in range(n-1,0,-1): # reversed
        d[i-1] -= (d[i] * -0.5) / b[i]

    for i in range(n):
        d[i] /= b[i]

    return np.array(d)