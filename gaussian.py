import numpy as np
import scipy.sparse as sparse
import scipy.special as special


class Gaussian(object):
    """This object generates a weighted average operator for the deflection of 
    the laminate cantilever. The weighting function is the Gaussian function.
    The weighted average deflection can be used to normalise the mode shape
    when performing calculations. The Gaussian function is chosen because
    it is differentiable and which allows this operator to be employed in 
    optimization problems to be solved with a gradient based optimization 
    algorithms. If the sigma parameters is chosen to be small, the weighted
    average deflection will be close to the deflection at the location of the 
    tip. 'xtip', 'ytip', and 'sigma' are given in micrometres.
    """
    def __init__(self, mesh, xtip, ytip, sigma):

        self._elements = mesh.get_elements()
        self._xtip = xtip
        self._ytip = ytip
        self._sigma = sigma
        self._a = 1e6 * mesh.a  # distance in um
        self._b = 1e6 * mesh.b  # distance in um
        self._dimension = mesh.n_mdof
    
        # Initial values of the operator and sensitivity.
        self._gau = self._assemble(self._operator_element)
        self._dgau = self._assemble(self._sensitivity_element)
        
        
    def get_operator(self):
        return self._gau
    
    
    def get_sensitivity(self):
        return self._dgau
    
    
    def update_ytip(self, ytip):
        self._ytip = ytip
        self._gau = self._assemble(self._operator_element)
        self._dgau = self._assemble(self._sensitivity_element)
        
        
    def _assemble(self, element_func):
        
        num = 20 * len(self._elements)
        row = np.zeros(num)
        col = np.zeros(num) 
        val = np.zeros(num)
        ntriplet = 0
        
        for e in self._elements:
            boundary = e.get_mechanical_boundary()
            dof = e.get_mechanical_dof()
            ge = element_func(e)
            for ii in range(20):
                if boundary[ii] is False:
                        row[ntriplet] = 0
                        col[ntriplet] = dof[ii]
                        val[ntriplet] = ge[0, ii]
                        ntriplet += 1
        
        shape = (1, self._dimension)
        gau = sparse.coo_matrix((val, (row, col)), shape=shape)
        return gau 
    
    
    def _operator_element(self, element):
                
        k1 = 1 / (4 * np.pi)
        k11 = 1 / (np.sqrt(2) * self._sigma)
        xdif = 2 * self._a * element.x0 - self._xtip
        ydif = 2 * self._b * element.y0 - self._ytip
        xa = (xdif - self._a) * k11
        xb = (xdif + self._a) * k11
        ya = (ydif - self._b) * k11
        yb = (ydif + self._b) * k11
        
        int1 = np.sqrt(np.pi/4) * (special.erf(xb) - special.erf(xa))
        int2 = self._sigma / np.sqrt(2) * (np.exp(-xb * xb) - np.exp(-xa * xa))
        int3 = np.sqrt(np.pi/4) * (special.erf(yb) - special.erf(ya))
        int4 = self._sigma / np.sqrt(2) * (np.exp(-yb * yb) - np.exp(-ya * ya))
        
        xi_a = [x/self._a for x in (-1, 1, 1, -1)]
        yi_b = [y/self._b for y in (-1, -1, 1, 1)]
        ge = [0 for i in range(4)]
        for i in range(4):
            k2 = (1 - xi_a[i] * xdif) * int1 - xi_a[i] * int2
            k3 = (1 - yi_b[i] * ydif) * int3 - yi_b[i] * int4
            ge[i] = k1 * k2 * k3

        ge = np.array([[0, 0, ge[0], 0, 0, 0, 0, ge[1], 0, 0, 
                        0, 0, ge[2], 0, 0, 0, 0, ge[3], 0, 0]])
        return ge
    
    
    def _sensitivity_element(self, element):
        
        k1 = 1 / (4 * np.pi)
        k11 = 1 / (np.sqrt(2) * self._sigma)
        xdif = 2 * self._a * element.x0 - self._xtip
        ydif = 2 * self._b * element.y0 - self._ytip
        xa = (xdif - self._a) * k11
        xb = (xdif + self._a) * k11
        ya = (ydif - self._b) * k11
        yb = (ydif + self._b) * k11
        
        int1 = np.sqrt(np.pi/4) * (special.erf(xb) - special.erf(xa))
        int2 = self._sigma / np.sqrt(2) * (np.exp(-xb * xb) - np.exp(-xa * xa))
        
        int5 = - k11 * (np.exp(-yb * yb) - np.exp(-ya * ya))
        int6 = np.sqrt(np.pi/4) * (special.erf(yb) - special.erf(ya))
        int7 = yb * np.exp(-yb * yb) - ya * np.exp(-ya * ya)
        
        xi_a = [x/self._a for x in (-1, 1, 1, -1)]
        yi_b = [y/self._b for y in (-1, -1, 1, 1)]
        ge = [0 for i in range(4)]
        for i in range(4):
            k2 = (1 - xi_a[i] * xdif) * int1 - xi_a[i] * int2
            dk3 = (1 - yi_b[i] * ydif) * int5 + yi_b[i] * int6 - yi_b[i] * int7
            ge[i] = k1 * k2 * dk3

        ge = np.array([[0, 0, ge[0], 0, 0, 0, 0, ge[1], 0, 0, 
                        0, 0, ge[2], 0, 0, 0, 0, ge[3], 0, 0]])
        return ge
    
    
    