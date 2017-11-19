import unittest
import numpy as np

from cantilevers import Cantilever
from laminate_dof import LaminateDOF
from mesh_v2 import UniformMesh


class TestCantilever(Cantilever):
    
    def __init__(self):
        
        a = 5e-6
        b = 5e-6
        topology = np.ones((3, 4))
        xtip = 25
        ytip = 55
        densities = np.ones((3, 4))
        name = 'LaminateDOF Test Cantilever'
        
        super().__init__(topology, a, b, xtip, ytip, densities, name)


class TestSymmetry(unittest.TestCase):
    
    def test_dofs(self):

        cantilever = TestCantilever()
        mesh = UniformMesh(cantilever)
        dof = LaminateDOF(mesh)
        
        print(dof.all_dofs)
        print(dof.fixed_dofs)
        print(dof.free_dofs)
        
                
if __name__ == '__main__':
    unittest.main()
