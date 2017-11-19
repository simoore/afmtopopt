import unittest
import time
import numpy as np
from collections import namedtuple
from materials import PiezoMumpsMaterial
from laminate_fem import LaminateFEM
from gaussian import Gaussian
from cantilevers import BJNCantilever, Cantilever


Element = namedtuple('Element', 'x0 y0')


class TestCantilever(Cantilever):
    def __init__(self):
        a = 5e-6
        b = 5e-6
        topology = np.ones((1, 1))
        densities = np.ones((1, 1))
        name = 'Unit Cantilever'
        xtip = 22
        ytip = 38
        
        super().__init__(topology, a, b, xtip, ytip, densities, name)


class TestGaussian(unittest.TestCase):
    
    def test_time(self):
        
        material = PiezoMumpsMaterial()
        bjn_cantilever = BJNCantilever()
        fem = LaminateFEM(bjn_cantilever, material)
        t0 = time.time()
        fem = LaminateFEM(bjn_cantilever, material)
        t1 = time.time()
        Gaussian(fem, bjn_cantilever, 0.1)
        t2 = time.time()
        print()
        print('The time to initialise FEM is %g seconds' % (t1 - t0))
        print('The time to initilaise Gau is %g seconds' % (t2 - t1))
        
        
    def test_operator_element(self):
        x0 = 2.5
        y0 = 3.5
        
        material = PiezoMumpsMaterial()
        cantilever = TestCantilever()
        fem = LaminateFEM(cantilever, material)
        #gau = Gaussian(fem.get_mesh(), 22, 38, 0.1)
        gau = Gaussian(fem, cantilever, 0.1)
        
        fields = (2e6 * cantilever.a * x0, 2e6 * cantilever.b * y0)
        print('Element origin (um): %g x %g' % fields)
        ElementInner = namedtuple('ElementInner', 'i j')
        ElementOuter = namedtuple('ElementOuter', 'element')
        ei = ElementInner(i=2, j=3)
        eo = ElementOuter(element=ei)
        ge = gau._operator_element(eo)
        ge_known = np.array([0, 0, 0.16, 0, 0, 0, 0, 0.04, 0, 0, 
                             0, 0, 0.16, 0, 0, 0, 0, 0.64, 0, 0])
        self.assertTrue(np.allclose(ge_known, ge))
        
        
        
    def test_sentivity_element(self):
        x0 = 2.5
        y0 = 3.5
        
        material = PiezoMumpsMaterial()
        cantilever = TestCantilever()
        fem = LaminateFEM(cantilever, material)
        gau = Gaussian(fem, cantilever, 0.1)
        
        fields = (2e6 * cantilever.a * x0, 2e6 * cantilever.b * y0)
        print('Element origin (um): %g x %g' % fields)
        ElementInner = namedtuple('ElementInner', 'i j')
        ElementOuter = namedtuple('ElementOuter', 'element')
        ei = ElementInner(i=2, j=3)
        eo = ElementOuter(element=ei)
        ge = gau._sensitivity_element(eo)
        ge = np.squeeze(ge)
        self.assertTrue(ge[2] < 0)
        self.assertTrue(ge[7] < 0)
        self.assertTrue(ge[12] > 0)
        self.assertTrue(ge[17] > 0)
        self.assertTrue(ge[2] < ge[7])
        self.assertTrue(ge[17] > ge[12])
    
    
if __name__ == '__main__':
    unittest.main()
