import unittest
import numpy as np
import cantilevers
import mesh_v2
import connectivity
import center_value
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class UnconnectedCantilever(cantilevers.Cantilever):
    
    def __init__(self):
        
        a = 10e-6
        b = 10e-6
        nelx = 20
        nely = 20
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
    
    def test_fem(self):
        
        cantilever = UnconnectedCantilever()
        mesh = mesh_v2.UniformMesh(cantilever) 
        fem = connectivity.Connectivity(mesh)
        fem.execute(mesh.get_densities())

        nelx, nely = cantilever.topology.shape
        d = np.zeros((nelx, nely))
        for e in fem._dof.dof_elements:
            d[e.element.i, e.element.j] = e.element.density
        
        z = np.zeros((nelx + 1, nely + 1))
        for n in fem._dof.dof_nodes:
            z[n.node.i, n.node.j] = fem.uall[n.dof]

        fig, ax = plt.subplots()
        ax.imshow(np.rot90(d), extent=(0, 1, 0, 1))
        
        fig, ax = plt.subplots()
        ax.imshow(np.rot90(z), interpolation='bilinear', 
                  cmap=cm.gray, extent=(0, 1, 0, 1))
        
        # Test the CenterValue operator produces a similar plot to the
        # one above.
        cv = center_value.CenterValue(fem._dof)
        h = cv.execute(fem.uall)
        hh = np.zeros((nelx, nely))
        for he, e in zip(h, fem._dof.dof_elements):
            hh[e.element.i, e.element.j] = he
            
        fig, ax = plt.subplots()
        ax.imshow(np.rot90(hh), cmap=cm.gray, extent=(0, 1, 0, 1))
        
        
if __name__ == '__main__':
    unittest.main()
