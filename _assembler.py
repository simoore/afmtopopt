import numpy as np
import scipy.sparse as sparse
from mesh import Element


class Assembler(object):
    def __init__(self, matrix, rows, cols, scale_func, 
                 row_boundary_func, col_boundary_func, row_dof_func, 
                 col_dof_func):
        
        self._matrix = matrix
        self._rows = rows
        self._cols = cols
        self._scale = scale_func
        self._row_boundary = row_boundary_func
        self._col_boundary = col_boundary_func
        self._row_dof = row_dof_func
        self._col_dof = col_dof_func
        
        nr, nc = matrix.shape
        
        self._nr = nr
        self._nc = nc
        
        
    def assemble(self, elements):
        """Returns the assemble matrix as defined by the assembler in a sparse 
        COO format.
        
        Returns
        -------
        mat : scipy.sparse.coo_matrix
        """
        
        self._num = self._nr * self._nc * len(elements)
        row = np.zeros(self._num)
        col = np.zeros(self._num)
        val = np.zeros(self._num)
        ntriplet = 0
        index = list(np.ndindex(self._nr, self._nc))
        for e in elements:
            row_bound = self._row_boundary(e)
            col_bound = self._col_boundary(e)
            row_dof = self._row_dof(e)
            col_dof = self._col_dof(e)
            #for ii in range(self._nr):
            #    for jj in range(self._nc):
            for ii, jj in index:
                if row_bound[ii] is False and col_bound[jj] is False:
                    row[ntriplet] = row_dof[ii]
                    col[ntriplet] = col_dof[jj]
                    val[ntriplet] = self._scale(e) * self._matrix[ii, jj]
                    ntriplet += 1
        
        shape = (self._rows, self._cols)
        mat = sparse.coo_matrix((val, (row, col)), shape=shape)
        return mat
 
    
class MassAssembler(Assembler):
    
    def __init__(self, matrix, rows, cols, scale_func):
        
        mbdy = Element.get_mechanical_boundary
        mdof = Element.get_mechanical_dof
        super().__init__(matrix, rows, cols, scale_func, mbdy, mbdy, mdof, mdof) 
        
        
class StiffnessAssembler(Assembler):
    
    def __init__(self, matrix, rows, cols, scale_func):
        
        mbdy = Element.get_mechanical_boundary
        mdof = Element.get_mechanical_dof
        super().__init__(matrix, rows, cols, scale_func, mbdy, mbdy, mdof, mdof) 
        
        
class PiezoAssembler(Assembler):
    
    def __init__(self, matrix, rows, cols, scale_func):
        
        mbdy = Element.get_mechanical_boundary
        ebdy = Element.get_electrical_boundary
        mdof = Element.get_mechanical_dof
        edof = Element.get_electrical_dof
        super().__init__(matrix, rows, cols, scale_func, mbdy, ebdy, mdof, edof) 
        
        
class CapAssembler(Assembler):
    
    def __init__(self, matrix, rows, cols, scale_func):
        
        ebdy = Element.get_electrical_boundary
        edof = Element.get_electrical_dof
        super().__init__(matrix, rows, cols, scale_func, ebdy, ebdy, edof, edof) 
