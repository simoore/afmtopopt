import unittest
from cantilevers import RectangularCantilever
from mesh import UniformMesh

class TestUniformMeshClass(unittest.TestCase):
    
    def setUp(self):
        self.nelx = 3
        self.nely = 4
        a = 1e-6
        b = 1e-6
        rect_cantilever = RectangularCantilever(a, b, self.nelx, self.nely)
        self.mesh = UniformMesh(rect_cantilever, 'plate')
        
        
    def test_number_of_dofs(self):
        node_total = (self.nelx + 1) * (self.nely + 1)
        n_dof = 3 * (node_total - self.nelx - 1)
        self.assertEqual(n_dof, self.mesh.n_mdof)
        self.assertEqual(node_total, self.mesh.n_node)
        
        
    def test_id_matches_order(self):
        elements = self.mesh.get_elements()
        ids = [e.id for e in elements]
        index = [i for i, _ in enumerate(elements)]
        self.assertTrue(ids == index)
        
        
if __name__ == '__main__':
    unittest.main()