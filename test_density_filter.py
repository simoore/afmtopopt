import unittest
from math import sqrt
import numpy as np
from cantilevers import RectangularCantilever
from mesh import UniformMesh
from density_filter import DensityFilter

class TestDensityFilter(unittest.TestCase):
    
    def test_operator(self):
        nelx = 3
        nely = 4
        a = 1e-6
        b = 1e-6
        rmin = 1.5
        fem_type = 'plate'
        
        rectangular_cantilever = RectangularCantilever(a, b, nelx, nely)
        mesh = UniformMesh(rectangular_cantilever, fem_type)
        density_filter = DensityFilter(mesh, rmin)
        
        diag = rmin - sqrt(2)
        row = [None for _ in range(12)]
        row[0] = np.array([1.5, 0.5, 0, 0, 0.5, diag, 0, 0, 0, 0, 0, 0])
        row[1] = np.array([0.5, 1.5, 0.5, 0, diag, 0.5, diag, 0, 0, 0, 0, 0])
        row[2] = np.array([0, 0.5, 1.5, 0.5, 0, diag, 0.5, diag, 0, 0, 0, 0])
        row[3] = np.array([0, 0, 0.5, 1.5, 0, 0, diag, 0.5, 0, 0, 0, 0])
        row[4] = np.array([0.5, diag, 0, 0, 1.5, 0.5, 0, 0, 0.5, diag, 0, 0])
        row[5] = np.array([diag, 0.5, diag, 0, 0.5, 1.5, 0.5, 0, diag, 0.5, diag, 0])
        row[6] = np.array([0, diag, 0.5, diag, 0, 0.5, 1.5, 0.5, 0, diag, 0.5, diag])
        row[7] = np.array([0, 0, diag, 0.5, 0, 0, 0.5, 1.5, 0, 0, diag, 0.5])
        row[8] = np.array([0, 0, 0, 0, 0.5, diag, 0, 0, 1.5, 0.5, 0, 0])
        row[9] = np.array([0, 0, 0, 0, diag, 0.5, diag, 0, 0.5, 1.5, 0.5, 0])
        row[10] = np.array([0, 0, 0, 0, 0, diag, 0.5, diag, 0, 0.5, 1.5, 0.5])
        row[11] = np.array([0, 0, 0, 0, 0, 0, diag, 0.5, 0, 0, 0.5, 1.5])
        
        operator = density_filter._operator.toarray()
        
        for i in range(12):
            actual_row = np.squeeze(operator[i,:])
            row[i] = row[i] / sum(row[i])
            self.assertTrue(np.all(np.isclose(actual_row, row[i])))
            
    
    
if __name__ == '__main__':
    unittest.main()

