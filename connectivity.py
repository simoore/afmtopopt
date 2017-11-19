import numpy as np
import scipy.sparse as sparse
from poisson_dof import PoissonDOF
from center_value import CenterValue


class Connectivity(object):
    
    def __init__(self, mesh):
        
        self.dof = PoissonDOF(mesh)
        self.ke, self.fe = self._compute_element_model(mesh.a, mesh.b)
        
        self._mesh = mesh
        self._center_value = CenterValue(self.dof)
        
        def kr_term(e): return np.hstack([e.dofs for _ in range(4)])
        def kc_term(e): return np.hstack([d*np.ones(4) for d in e.dofs]) 
        kr = np.hstack([kr_term(e) for e in self.dof.dof_elements])
        kc = np.hstack([kc_term(e) for e in self.dof.dof_elements])
        fr = np.hstack([e.dofs for e in self.dof.dof_elements])
        fc = np.zeros(fr.shape)
        
        self._k_index = (kr, kc)
        self._f_index = (fr, fc)
        self._k_shape = (self.dof.n_dof, self.dof.n_dof)
        self._f_shape = (self.dof.n_dof, 1) 
        self.assemble(mesh.get_densities())
        
        
    def get_conduction_matrix(self, free=False):

        if free is False:
            return self.ktau
        return self.ktau[self.dof.free_dofs, :][:, self.dof.free_dofs]
    
    
    def get_heating_matrix(self, free=False):
        
        if free is False:
            return self.ftau
        return self.ftau[self.dof.free_dofs, :]
    
    
    def get_tau_mu_transform(self, free=False):
        """This operator maps the temperature field (tau) to the connectivty 
        penalization parameters (mu). There are two versions, one for all 
        dofs in the temperature field (tau), and one for only the free DOFs in 
        the termperature field (tau).
        """
        transform = self._center_value.sensitivity()
        if free is False:
            return transform
        return transform[:, self.dof.free_dofs]
    
    
    def get_connectivity_penalty(self, tau):
        """For a given temperature field (tau), this function returns the the 
        connectivity penalty value for each element.
        """
        mu = self._center_value.execute(tau)
        return mu
        
    
    def thermal_analysis(self):
        """Applies the boundary conditions onto the system matrices (ktau) 
        and (ftau). Computes the solution of the equation Ku=f. Reinserts the
        boundary DOFs into the solution then returns.
        """
        sysk = self.ktau[self.dof.free_dofs, :][:, self.dof.free_dofs]
        sysf = self.ftau[self.dof.free_dofs, :]
        ufree = sparse.linalg.spsolve(sysk, sysf)
        uall = np.zeros(self.dof.all_dofs.shape)
        uall[self.dof.free_dofs] = ufree
        return uall, ufree
    
        
    def assemble(self, xs):

        if xs is None:
            return
        
        k, q = self._set_penalty(xs)
        kexpand = np.expand_dims(k, axis=1)
        fexpand = np.expand_dims(q, axis=1)
        kv = np.ravel(kexpand @ np.expand_dims(self.ke.ravel(), axis=0))
        fv = np.ravel(fexpand @ self.fe.T)
        
        sysk = sparse.coo_matrix((kv, self._k_index), shape=self._k_shape)
        sysf = sparse.coo_matrix((fv, self._f_index), shape=self._f_shape)
        self.ktau = sysk.tocsr()
        self.ftau = sysf.tocsr()
    
    
    def _set_penalty(self, xs):
        
        q0, k0, eps = 1, 1e4, 1e-4
        k = k0*xs + (1 - xs)*eps*k0
        q = q0*xs
        self.thermal_grad = (k0 - eps*k0)*np.ones_like(xs)
        self.heat_grad = q0*np.ones_like(xs)
        return k, q


    def _compute_element_model(self, a, b):
        """
        Parameters
        ----------
        a: Half the width of an element in the x-direction.
        b: Half the width of an element in the y-direction.
        """
        points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / np.sqrt(3)
        jacobian = a*b
        ke = np.zeros((4, 4))
        fe = np.zeros((4, 1))
        for p in points:
            n, dndxi, dndeta = self._shapes(p)
            ke += jacobian * (dndxi.T @ dndxi / a*a + dndeta.T @ dndeta / b*b)
            fe += jacobian * n.T
        return ke, fe
        
          
    @staticmethod
    def _shapes(point):
        """
        Parameters
        ----------
        point: the point (xi, eta) to evaluate the shape functions.
        """
        xi, eta = point
        xs = np.array([[-1, 1, 1, -1]])
        es = np.array([[-1, -1, 1, 1]])
        n = 0.25 * (1 + xs * xi) * (1 + es * eta)
        dndxi = xs * 0.25 * (1 + es * eta)
        dndeta = es * 0.25 * (1 + xs * xi) 
        return n, dndxi, dndeta
    