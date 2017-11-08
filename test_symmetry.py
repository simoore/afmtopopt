from math import floor
import unittest
import numpy as np
import cantilevers
import materials
import finite_element
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
        fem = finite_element.LaminateFEM(cantilever, material)
        sym = symmetry.Symmetry(fem)
        
        # print(sym._operator)
        
        design_vars = np.arange(0, 18)
        densities = sym.execute(design_vars)
        topology = np.empty((fem.mesh.nelx, fem.mesh.nely))
        for e, d in zip(fem.mesh.elements, densities):
            topology[floor(e.x0), floor(e.y0)] = d
            
        # print(topology)
        
        nx, ny = topology.shape
        for x in range(nx):
            for y in range(ny):
                self.assertTrue(topology[x, y] == topology[nx - x - 1, y])
                
                
    def test_initial(self):
        
        material = materials.PiezoMumpsMaterial()
        cantilever = TestCantilever()
        fem = finite_element.LaminateFEM(cantilever, material)
        sym = symmetry.Symmetry(fem)
        x0 = sym.initial(fem.get_element_densities())
        
        #print(x0)
        
        self.assertTrue(np.all(x0 == 1))
        
        
if __name__ == '__main__':
    unittest.main()
