import sys

module_path = '..\\..'
if module_path not in sys.path:
    sys.path.append(module_path)
    
import unittest
from afmtopopt.laminate_model import LaminateModel
from afmtopopt.materials import PiezoMumpsMaterial


class TestLaminateModel(unittest.TestCase):

    def test_kvv(self):
        # This test the value of the matrix kvv is equal to the parallel plate
        # capacitance of the structure. Assuming one patch over the entire 
        # cantilever. The constants here should match those in the 
        # PiezoMumpsMaterialObject.
        
        nelx = 3
        nely = 4
        a = 1e-6
        b = 1e-6
        
        material = PiezoMumpsMaterial()
        topology = RectangularCantilever(a, b, nelx, nely)
        self.fem_a = LaminateFEM(topology, material)
        self.area = nelx * nely * 4 * a * b
        
        perm_free = 8.85418782e-12
        thickness = 0.5e-6
        epsilon = perm_free * 10.2
        capacitance = self.area * epsilon / thickness
        kvv = self.fem_a.kvv
        self.assertEqual(capacitance, kvv.A[0, 0])
        
    
if __name__ == '__main__':
    unittest.main()
