import unittest
import numpy as np
import cantilevers
import materials
from laminate_fem import LaminateFEM
import symmetry


class TestCantilever(cantilevers.Cantilever):
    
    def __init__(self):
        
        a = 5e-6
        b = 5e-6
        topology = np.ones((5, 6))
        xtip = 25
        ytip = 55
        densities = np.ones((5, 6))
        name = 'Symmetry Test Cantilever'
        
        super().__init__(topology, a, b, xtip, ytip, densities, name)


class TestSymmetry(unittest.TestCase):
    
    def test_operator(self):

        material = materials.PiezoMumpsMaterial()
        cantilever = TestCantilever()
        fem = LaminateFEM(cantilever, material)
        sym = symmetry.Symmetry(fem)
        
        # print(sym._operator)
        
        nelx, nely = fem.cantilever.topology.shape
        design_vars = np.arange(0, 18)
        densities = sym.execute(design_vars)
        topology = np.empty((nelx, nely))
        for e, d in zip(fem.mesh.elements, densities):
            topology[e.i, e.j] = d
            
        # print(topology)
        
        nx, ny = topology.shape
        for x in range(nx):
            for y in range(ny):
                self.assertTrue(topology[x, y] == topology[nx - x - 1, y])
                
                
    def test_initial(self):
        
        material = materials.PiezoMumpsMaterial()
        cantilever = TestCantilever()
        fem = LaminateFEM(cantilever, material)
        sym = symmetry.Symmetry(fem)
        x0 = sym.initial(fem.mesh.get_densities())
        
        #print(x0)
        
        self.assertTrue(np.all(x0 == 1))
        
        
if __name__ == '__main__':
    unittest.main()
