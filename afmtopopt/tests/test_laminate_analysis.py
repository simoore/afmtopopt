import numpy as np
from ..materials import PiezoMumpsMaterial
from ..cantilevers import InitialCantileverHigherFreq, Cantilever
from ..laminate_analysis import LaminateAnalysis

class TestCantileverA(Cantilever):
    
    def __init__(self):
        
        a = 5e-6
        b = 5e-6
        topology = np.ones((2, 2))
        xtip = 10
        ytip = 18
        densities = np.ones((2, 2))
        name = 'Analysis Test Cantilever A'
        
        super().__init__(topology, a, b, xtip, ytip, densities, name)

material = PiezoMumpsMaterial()
cantilever = TestCantileverA()
analysis = LaminateAnalysis(cantilever, material, 0.1, False)
analysis.execute_analysis()
