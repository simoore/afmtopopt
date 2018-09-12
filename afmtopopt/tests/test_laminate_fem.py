import unittest
import time
import numpy as np
from laminate_fem import LaminateFEM
from laminate_model import LaminateModel
from cantilevers import RectangularCantilever, BJNCantilever
from materials import PiezoMumpsMaterial
import materials, cantilevers, analysers
from gaussian import Gaussian
#import _plate_materials
#import _plate_fem


class SingleLayerSoiMumpsMaterial(materials.LaminateMaterial):
    
    def __init__(self):
        perm_free = 8.85418782e-12
        
        si_h = 10e-6
        si_rho = 2330
        si_elastic = 169e9
        si_nu = 0.064
        # si_nu = 0.29
        aln_perm = perm_free * 10.2
        aln_piezo = 2e-12
        
        si = materials.LaminateLayer(si_h, si_rho, si_elastic, si_nu, 
                                     aln_perm, aln_piezo)
        layers = (si,)
        super().__init__(layers, si)
        
        
class ThickSoiMumpsMaterial(materials.LaminateMaterial):
    
    def __init__(self):
        perm_free = 8.85418782e-12
        
        si_h = 11.5e-6
        si_rho = 2330
        si_elastic = 169e9
        si_nu = 0.064
        # si_nu = 0.29
        aln_perm = perm_free * 10.2
        aln_piezo = 2e-12
        
        si = materials.LaminateLayer(si_h, si_rho, si_elastic, si_nu, 
                                     aln_perm, aln_piezo)
        layers = (si,)
        super().__init__(layers, si)
        
        
class DoubleLayerSoiMumpsMaterial(materials.LaminateMaterial):
    
    def __init__(self):
        perm_free = 8.85418782e-12
        
        #si_h = 10e-6
        si_rho = 2330
        si_elastic = 169e9
        si_nu = 0.064
        # si_nu = 0.29
        aln_perm = perm_free * 10.2
        aln_piezo = 2e-12
        
        sia = materials.LaminateLayer(8.5e-6, si_rho, si_elastic, si_nu, 
                                      aln_perm, aln_piezo)
        sib = materials.LaminateLayer(1.5e-6, si_rho, si_elastic, si_nu, 
                                      aln_perm, aln_piezo)
        layers = (sia, sib)
        super().__init__(layers, sia)
        
        
class TestLaminateFEM(unittest.TestCase):
    
#    def test_single_layer(self):
#        # Using the plate finite element, and the laminate finite element
#        # with a single layer, the mechanical properties should be indentical.
#        # Modal analysis is performed for both FEM formulation and the 
#        # eigenvalues are compared.
#        
#        material = SingleLayerSoiMumpsMaterial()
#        cantilever = BJNCantilever()
#        fem = LaminateFEM(cantilever, material)
#        wl, vl = fem.modal_analysis(4)
#
#        material = _plate_materials.SoiMumpsMaterial()
#        fem = _plate_fem.PlateFEM(cantilever, material)
#        wp, vp = fem.modal_analysis(4)
#        
#        self.assertTrue(np.all(np.isclose(wl, wp)))
#        
#        material = DoubleLayerSoiMumpsMaterial()
#        fem = LaminateFEM(cantilever, material)
#        wz, vz = fem.modal_analysis(4)
#        
#        self.assertTrue(np.all(np.isclose(wz, wp)))
        
        
    def test_compare_plate_10um_and_11_5um(self):
        
        cantilever = cantilevers.BJNCantilever()
        
        print()
        print('The first material is the 10um SOI for comparison')
        
        material = SingleLayerSoiMumpsMaterial()
        fem = LaminateFEM(cantilever, material)
        analyser = analysers.CantileverAnalyser(fem)
        analyser.identify_modal_parameters()
        
        print()
        print('SOIMumps single layer of 11.5um') 
        
        material = ThickSoiMumpsMaterial()
        fem = LaminateFEM(cantilever, material)
        analyser = analysers.CantileverAnalyser(fem)
        analyser.identify_modal_parameters()
        
        print()
        print('Piezoelectric material')
        
        material = PiezoMumpsMaterial()
        fem = LaminateFEM(cantilever, material)
        analyser = analysers.CantileverAnalyser(fem)
        analyser.identify_modal_parameters()
        
        
    
    def test_timing(self):
        # Give an indication of the computation time of the finite element
        # analysis.
        
        material = PiezoMumpsMaterial()
        cantilever = BJNCantilever()
        fem = LaminateFEM(cantilever, material)
        dens = np.ones(fem.mesh.n_elem)
        lam, phi = fem.modal_analysis(1)
        kuu = fem.kuu
        kuv = fem.kuv
        gaussian = Gaussian(fem, cantilever, 0.1)
        guu = gaussian.get_operator()
        wtip = np.asscalar(guu @ phi)
        charge = np.asscalar(kuv.T @ phi)
        k1 = np.asscalar(phi.T @ kuu @ phi / wtip ** 2)
        
        t0 = time.time()
        fem.set_penalty(dens)
        fem.assemble()
        t1 = time.time()
        fem.charge_grad(lam[0], phi, wtip, charge, guu)
        t2 = time.time()
        fem.stiff_grad(lam[0], phi, wtip, k1, guu)
        t3 = time.time()
        print()
        print('Update Element Densities Time %g' % (t1 - t0))
        print('Charge Sensitivity Time       %g' % (t2 - t1))
        print('Stiffness Sensitivty Time     %g' % (t3 - t2))
        
        
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
        
            
    def test_zero_terms(self):
        # In the derivation of the stiffness matrix, it is assumed a couple of
        # terms are zero. This checks this result numercially.
        
        material = PiezoMumpsMaterial()
        elements = LaminateModel(material, 1e-6, 1e-6)
        bs1, bs2, bs3 = elements._dofs_to_strain_matrix((0.5, 0.5))
        c = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0], 
                      [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        term1 = bs1.T @ c @ bs3
        term2 = bs2.T @ c @ bs3
        self.assertTrue(np.all(term1 == 0))
        self.assertTrue(np.all(term2 == 0))
        
        
    def test_charge_grad(self):
        pass
        
        
if __name__ == '__main__':
    unittest.main()