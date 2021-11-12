import numpy as np

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
        
        def tridiag_solve(b, d): # specialization with a = c = -0.5
            n = len(d)
            for i in range(n-1):
                frac = -0.5/b[i]
                d[i+1] -= d[i] * frac
                b[i+1] -=  -0.5 * frac
            for i in range(n-1,0,-1): # reversed
                d[i-1] -= (d[i] * -0.5) / b[i]
            return d / b
        
        if(self._mu == 0.): 
            return 0.5*nabla # handle the non-penalized case
        inv_diag = (1./self._mu) * inverse_hessian_h(np.diff(x), self._eps)        
        temp = tridiag_solve(1. + inv_diag, np.diff(nabla))
        return 0.5*nabla - 0.25*self._A_top(temp)

    @staticmethod
    def _A_top(z): # apply transpose of diff matrix A to z
        return np.r_[0., z] - np.r_[z, 0.] # [-z[0],-np.diff(z),z[-1]]
#