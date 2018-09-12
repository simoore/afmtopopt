import numpy as np
import scipy.sparse as sparse


class CenterValue(object):
    """
    Note that this operator applies to solutions of Poisson's equation 
    coded in connectivity.py. It interpolates the solution to the equation
    to return the value at the center point of each element.
    """
    def __init__(self, dof):

        row = np.array([e.index for e in dof.mesh.elements for _ in range(4)])
        col = np.array([n.dof for e in dof.dof_elements for n in e.dof_nodes])
        val = 0.25 * np.ones(4 * dof.mesh.n_elem)
            
        shape = (dof.mesh.n_elem, dof.n_dof)
        operator = sparse.coo_matrix((val, (row, col)), shape=shape)
        self._operator = operator.tocsr()
        
        
    def execute(self, xs):
        """The input 'xs' are all the dofs in the system. It returns a value
        for each element.
        """
        return self._operator @ xs
    
    
    def sensitivity(self):
        """This returns the jacobian of this operator.
        """
        return self._operator
