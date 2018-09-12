import unittest
import numpy as np
import cantilevers
import mesh_v2
from poisson_dof import PoissonDOF
import center_value


class UnconnectedCantilever(cantilevers.Cantilever):
    
    def __init__(self):
        
        a = 10e-6
        b = 10e-6
        nelx = 3
        nely = 4
        xtip = 1e6 * (a * nelx)
        ytip = 1e6 * (2 * nely * b - b)
        
        topology = np.ones((nelx, nely))
        
        bot = np.ones((20, 5))
        mid = np.vstack((1e-3 * np.ones((15, 10)), np.ones((5, 10))))
        top = np.vstack((np.ones((10, 5)), 1e-3 * np.ones((5, 5)), np.ones((5, 5))))
        densities = np.hstack((bot, mid, top))
        
        name = 'Unconnected Cantilever'
        
        super().__init__(topology, a, b, xtip, ytip, densities, name)


class TestSymmetry(unittest.TestCase):
    
    def test_operator(self):
        
        cantilever = UnconnectedCantilever()
        mesh = mesh_v2.UniformMesh(cantilever) 
        dof = PoissonDOF(mesh)
        center = center_value.CenterValue(dof)
        
        dofs = np.arange(dof.n_dof)
        val = center.execute(dofs)
        
        for e in dof.dof_elements:
            t1 = val[e.element.index]
            t2 = 0.25 * sum([dofs[n.index] for n in e.element.nodes])
            self.assertEqual(t1, t2)
        
        
        
        
if __name__ == '__main__':
    unittest.main()
