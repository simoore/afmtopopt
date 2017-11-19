from math import ceil
import numpy as np
import scipy.sparse as sparse


class Symmetry(object):
    """
    Public Attributes
    -----------------
    :self.dimension: The number of design variables.
    """
    def __init__(self, fem):

        nelx, nely = fem.cantilever.topology.shape
        ks = ceil(nelx / 2)
        col = np.empty(fem.mesh.n_elem)
        for i, e in enumerate(fem.mesh.elements):
            x = e.i
            y = e.j
            xsym = x if (e.i + 0.1) < ks else nelx - x - 1
            col[i] = round(ks*y + xsym)
        row = np.arange(0, fem.mesh.n_elem)
        val = np.ones(fem.mesh.n_elem)
        self.dimension = round(ks * nely)
        shape = (fem.mesh.n_elem, self.dimension)
        self._operator = sparse.coo_matrix((val, (row, col)), shape=shape)
        self._operator = self._operator.tocsr()
        
        
    def initial(self, xs):
        """The input 'xs' are the initial pseudo-densities and returned are
        the initial design parameters. The average value of the 
        pseudo-densities associated with each design variable is returned.
        """
        scale = np.squeeze(np.sum(self._operator, 0).A)
        return (self._operator.T @ xs) / scale
    
        
    def execute(self, xs):
        """The input 'xs' are the design parameters and it returns the 
        pseudo-densities of each element.
        """
        return self._operator @ xs
    
    
    def sensitivity(self):
        """This returns the jacobian of this symmetry operator, which is 
        a sparse matrix.
        """
        return self._operator