"""
Module containing relatively simple methods for trend estimation.
Compute an estimate of the underlying trend of a timeseries realization y.
Timeseries y typically contains noise, outliers/spikes, high-frequency components, etc.
"""

import numpy as np
from scipy.linalg import solve_banded
#import utils
import utils_cython as utils


class ATV_denoise():
    """ 
    Approximate TV denoising tool (Newton style)
    psi(x) = f(x) + h(x), both twice differentiable
    Efficient evaluation of gradient and Hessian
    Adaptive selection of regularization parameter
    """
    def __init__(self, eps=1e-3, alpha=0.01, beta=0.5):
        assert 0. < eps
        assert 0. < alpha < 1.
        assert 0. < beta < 1.
        self._eps = eps     # approximation handle
        self._alpha = alpha # line search param
        self._beta = beta   # line search param
        self._verbose = True
        
    def denoise(self, y, mu=None):
        """ Generate denoised version x of corrupted input signal y """
        n = len(y)
        x = np.zeros(n) + y.mean() # 
        nabla = np.zeros(n) # gradient
        delta = np.zeros(n) # inverse Hessian dot gradient, H.I.dot(nabla)
        if(mu is not None):
            assert 0. <= mu, "Regularization parameter must be non-negative"
            self._mu = mu # provided by user
        else:
            TV = np.abs(np.diff(y)).sum()
            self._mu = 2.*(n*np.var(y)/TV) # adaptive
        
        for k in range(50):
            nabla[:] = self._gradient(x, y)
            delta[:] = self._H_inv_dot_gradient(x, nabla)
            lambda_sq = nabla.dot(delta) # quadratic form                
            t = 1. # backtracking line search
            cost_xy = self._cost(x,y)
            while self._cost(x-t*delta,y) > cost_xy - self._alpha*t*lambda_sq:
                t *= self._beta
            x[:] -= t*delta # update
            if( abs(t*np.abs(delta).max()) < 1e-9 ):
                break            
            if(self._verbose == True):
                print(f'k={k} t={t} lambda={np.sqrt(lambda_sq)}')
        
        return x        

    def _cost(self, x, y):
        """ Eval cost function of approx. TV de-noising problem """
        mu, eps = self._mu, self._eps
        r = x - y # residual
        dx = np.diff(x)
        cost = r.dot(r) - mu * float(len(x)-1) * eps
        cost += mu * (np.sqrt(dx*dx + eps*eps)).sum()
        return cost
    
    def _gradient(self, x, y):
        """ Eval derivative of cost function """
        def nabla_h(z, epsilon): # helper
            return z/np.sqrt(z*z + epsilon*epsilon)
        mu, eps = self._mu, self._eps
        return 2.*(x-y) + mu*self._A_top(nabla_h(np.diff(x), eps))

    def _H_inv_dot_gradient(self, x, nabla):
        """ 
        Eval u = H.I.dot(v) without realizing H (n x n) 
        H = 2*I + mu*A.T.dot(D.dot(A)), A diff matrix, D diagonal
        Use inversion lemma and tridiag structure of A.dot(A.T)
        """
        def inverse_hessian_h(z, eps):
            temp_h = z*z + eps*eps
            return np.sqrt(temp_h) / (1.-((z*z)/temp_h))
        
        if(self._mu == 0.): 
            return 0.5*nabla # handle the non-penalized case
        inv_diag = (1./self._mu) * inverse_hessian_h(np.diff(x), self._eps)        
        temp = utils.tridiag_solve_specialcase(1. + inv_diag, np.diff(nabla))
        return 0.5*nabla - 0.25*self._A_top(temp)

    @staticmethod
    def _A_top(z): # apply transpose of diff matrix A to z
        return np.r_[0., z] - np.r_[z, 0.] # [-z[0],-np.diff(z),z[-1]]
#

class HodrickPrescottFiltering():
    """
    Reference: "$\ell_1$ Trend Filtering" paper (Steven Boyd)
    Naive and efficient routines for computation of HP trend estimates.
    Naive routines mainly for educational/learning purpose.
    Fixed-size methods and variable size methods (where n=len(y) is not known a priori).
    """

    def __init__(self, n=1000, dtype_='float32'):        
        # preallocation of the 5 banded diagonals (2 upper, main, 2 lower)
        # for use-cases with numerous/repeated calculations with fixed size n
        self.dtype = dtype_
        self.bands = self._prep_bands_for_fixed_size(n=n, dtype=dtype_)
    
    @staticmethod
    def _build_D_matrix(n: int, dtype='float32'):
        """
        Explicit realization of the (n-2) x n second-order diff matrix D.
        """
        assert (n >= 3), 'Dimension n too small'
        D = np.zeros([n-2,n], dtype=dtype)
        row = np.r_[1.,-2.,1.,np.zeros(n-3, dtype=dtype)]
        for i in range(n-2):
            D[i,:] = np.roll(row, i)
        
        return D

    @staticmethod
    def _build_DtopD_matrix(n: int, dtype='float32'):
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

    def _filter_naive(self, y: np.ndarray, lambda_=1e3):
        """
        Involves explicit realization of (n x n) matrix. Not feasible for large n.
        """
        n = len(y)    
        A = np.eye(n) + (2.*lambda_)*self._build_DtopD_matrix(n)
        x_hp = np.linalg.solve(A, y)
        
        return x_hp
    
    @staticmethod
    def _prep_bands_for_fixed_size(n: int, dtype='float32'):
        """
        See scipy.linalg.solve_banded() for terminology.
        Preallocation of (5 x n) array and prepping of as much as possible.
        REMARK: lambda scaling and diagonal addition not handled here.
        """
        
        # prep individual bands
        wipe = 0.
        upper2 = np.ones(n)
        upper2[0:2] = wipe
        upper1 = np.zeros(n) - 4.
        upper1[0] = wipe
        upper1[ 1] = -2.
        upper1[-1] = -2.
        diag = np.zeros(n) + 6.
        
        # arrange in array as required by scipy routine
        bands = np.zeros([2+1+2,n], dtype=dtype)
        bands[0,:] = upper2[:]
        bands[1,:] = upper1[:]
        bands[2,:] = diag[:]
        bands[3,:] = np.roll(upper1, -1) # lower1
        bands[4,:] = np.roll(upper2, -2) # lower2

        return bands

    def filter_fixed(self, y: np.ndarray, lambda_=1e3):
        """
        Solve banded linear system using SciPy routine. Fixed input size.
        The .copy() is a bit faster than calling _prep_bands_for_fixed_size() every time.
        """
        n = len(y)
        assert (self.bands.shape[1] == n), 'Not correct size, it is assumed fixed!'
        bands = self.bands.copy()
        bands[2,:] += (0.5/lambda_) # add scaled identity matrix
        x_hp = solve_banded((2,2), bands, (0.5/lambda_) * y, 
                            check_finite=False, overwrite_ab=True, overwrite_b=True)
        
        return x_hp


    def filter(self, y: np.ndarray, lambda_=1e3):
        """
        Solve banded linear system using SciPy routine. Any input size.
        """
        n = len(y)
        bands = self._prep_bands_for_fixed_size(n, dtype=self.dtype)
        bands[2,:] += (0.5/lambda_) # add scaled identity matrix
        x_hp = solve_banded((2,2), bands, (0.5/lambda_) * y,
                            check_finite=False, overwrite_ab=True, overwrite_b=True)
        
        return x_hp
