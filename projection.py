import numpy as np
import scipy.sparse as sparse


class Projection(object):
    
    def __init__(self, beta):
        self._beta = beta
        
        
    def execute(self, xs):
        """The input 'xs' are the puedo-densities.
        """
        ys = 1 - np.exp(-self._beta * xs) + xs * np.exp(-self._beta)
        return ys
    
    
    def sensitivity(self, xs):
        """This returns the Jacobian of this projection function, which due is 
        a sparse diagonal matrix.
        """
        shape = (len(xs), len(xs))
        dys = self._beta * np.exp(-self._beta * xs) + np.exp(-self._beta)
        mat = sparse.diags(dys, shape=shape)
        return mat
