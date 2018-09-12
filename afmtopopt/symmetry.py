from math import ceil
import numpy as np
import scipy.sparse as sparse


class Symmetry(object):

    def __init__(self, fem, original=False):

        if original is True:
            self._init_a(fem)
        else:
            self._init_b(fem)
        
    
    def _init_a(self, fem):
        """Only works for rectangular domains. Use _init_b in future.
        """
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
        self._dimension = round(ks * nely)
        shape = (fem.mesh.n_elem, self._dimension)
        self._operator = sparse.coo_matrix((val, (row, col)), shape=shape)
        self._operator = self._operator.tocsr()
        
        
    def _init_b(self, fem):

        nelx, nely = fem.cantilever.topology.shape
        ks = ceil(nelx / 2)
        col = np.empty(fem.mesh.n_elem)
        col_2D = np.zeros((nelx, nely))
        col_num = 0
        for i, e in enumerate(fem.mesh.elements):
            if e.i + 0.1 < ks:
                col_2D[e.i, e.j] = col_num
                col[i] = col_2D[e.i, e.j]
                col_num += 1
        for i, e in enumerate(fem.mesh.elements):
            if e.i + 0.1 >= ks:
                col_2D[e.i, e.j] = col_2D[nelx - e.i - 1, e.j]
                col[i] = col_2D[e.i, e.j]
        row = np.arange(0, fem.mesh.n_elem)
        val = np.ones(fem.mesh.n_elem)
        self._dimension = int(max(col) + 1)
        shape = (fem.mesh.n_elem, self._dimension)
        self._operator = sparse.coo_matrix((val, (row, col)), shape=shape)
        self._operator = self._operator.tocsr()
        
        
    @property
    def dimension(self):
        return self._dimension
        
        
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