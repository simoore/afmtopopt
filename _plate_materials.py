import numpy as np


class IsotropicPlateMaterial(object):
    
    def __init__(self, rho, e, nu, h):
        """
        Parameters
        ----------
        rho : float
        The density of the material in kg/m^3.
        
        e : float
        Elastic modulus in Pa.
        
        nu : float
        Poisson's ratio.
        
        h : float
        The thickness of the plate.
        """
        
        self.rho = rho                  # density
        self.e = e                      # elastic modulus
        self.nu = nu                    # poissons ratio
        self.kappa = np.pi ** 2 / 12    # shear corretion factor
        self.g = 0.5 * e / (1 + nu)     # shear modulus
        self.h = h                      # plate thickness
        
        self.CI = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        self.CI = e / (1 - nu * nu) * self.CI
        self.CO = np.diag((self.g, self.g))
        self.CM = np.diag((rho * h, rho * h ** 3 / 12, rho * h ** 3 / 12))


    def get_finite_element_parameters(self):
        return self.CI, self.CO, self.CM, self.h, self.kappa
    

class SoiMumpsMaterial(IsotropicPlateMaterial):
    
    def __init__(self):
        rho = 2330
        e = 169e9
        nu = 0.064 
        h = 10e-6
        super().__init__(rho, e, nu, h)
        