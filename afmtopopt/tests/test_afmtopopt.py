#from math import pi
#import numpy as np
#import unittest
#import time
#from afmtopopt import DensityFilter
#from materials import IsotropicPlateMaterial, PiezoMumpsMaterial
#from mesh import UniformMesh
#from cantilevers import RectangularCantilever, BJNCantilever
#from finite_element import LaminateFEM, PlateFEM
#from gaussian import Gaussian
#
#
#class TestIsotropicPlateMaterialClass(unittest.TestCase):
#    def setUp(self):
#        self.material = IsotropicPlateMaterial(2330, 169e9, 0.064, 10e-6)
# 
#    def test_get_finite_element_parameters(self):
#        CI, CO, CM, H, Kappa = self.material.get_finite_element_parameters()
#        self.assertEqual(pi ** 2 / 12, Kappa)
#        
#if __name__ == '__main__':
#    unittest.main()