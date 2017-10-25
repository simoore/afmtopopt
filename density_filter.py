from math import ceil, sqrt
import numpy as np
import scipy.sparse as sparse


class DensityFilter(object):
    
    def __init__(self, mesh, rmin):
        self._operator = self._create_operator(mesh, rmin)
        
    
    def _create_operator(self, mesh, rmin):
        
        elements = mesh.get_elements()
        
        offset = []
        for ii in range(ceil(-rmin), ceil(rmin)):
            y0 = sqrt(rmin ** 2 - ii ** 2)
            for jj in range(ceil(-y0), ceil(y0)):
                offset.append((ii, jj))
                
        neighbours = [[[] for _ in range(mesh.nely)] for _ in range(mesh.nelx)]
        for e in elements:
            for o in offset:
                ii = e.nodes[0].i + o[0]
                jj = e.nodes[0].j + o[1]
                if 0 <= ii < mesh.nelx and 0 <= jj < mesh.nely:
                    neighbours[ii][jj].append(e)
        
        num_triplets = len(elements) * (2 * (ceil(rmin) - 1) + 1)**2
        iH = np.zeros(num_triplets)
        jH = np.zeros(num_triplets)
        sH = np.zeros(num_triplets)
        k = 0
        
        def norm(e1, e2): 
            i1, j1 = e1.nodes[0].i, e1.nodes[0].j
            i2, j2 = e2.nodes[0].i, e2.nodes[0].j
            return sqrt((i1 - i2)**2 + (j1 - j2)**2)
        
        for e1 in elements:
            x = e1.nodes[0].i
            y = e1.nodes[0].j
            for e2 in neighbours[x][y]:
                coefficient = max(0, rmin - norm(e1, e2))
                if coefficient != 0:
                    iH[k] = e1.id
                    jH[k] = e2.id
                    sH[k] = coefficient
                    k += 1
                    
        shape = (len(elements), len(elements))
        Hn = sparse.coo_matrix((sH, (iH, jH)), shape=shape).tocsc()
        Hd = sparse.diags(1 / np.squeeze(np.sum(Hn, 1).A))
        H = Hd @ Hn
        return H
    
    
    def execute(self, xs):
        """If xs is one dimensional, the @ operator knows it is supposed to be
        a column vector. In this case, xf is one dimensional also.
        """
        xf = self._operator @ xs
        return xf
    
    
    def sensitivity(self):
        return self._operator
    