import numpy as np
import scipy.sparse as sparse

from laminate_fem import LaminateFEM
from gaussian import Gaussian
from connectivity import Connectivity


class LaminateAnalysis(object):
    
    def __init__(self, cantilever, material, to_connect, pmu=0.0):
        """
        :param cantilever: The cantilever object that defines the initial
            conditions of the topology optimization routine.
        :param to_connect: Indicates whether to apply the connectivity 
            penalizations.
        :param pmu: The connectivity penalty factor.
        """
        self.to_connect = to_connect
        self.fem = LaminateFEM(cantilever, material, to_connect, pmu)
        self.connectivity = Connectivity(self.fem.mesh)
        self.gaussian = Gaussian(self.fem, self.fem.cantilever, 0.1)
        self.mdofmat = self.prepare_mdof_matrix()
        self.tdofmat = self.prepare_tdof_matrix()


    def execute_analysis(self, dens=None):
        """
        :param dens: The new pseudo-densities of the cantilever. If it is None, 
            the finite element models are not re-assembled.
        """
        # Update the connectivity model and perform thermal analysis.
        self.connectivity.assemble(dens)
        tau, _ = self.connectivity.thermal_analysis()
    
        # Update piezoelectric laminate model and perform modal analysis.
        mu = self.connectivity.get_connectivity_penalty(tau)    
        self.fem.assemble(dens, mu)
        w, v = self.fem.modal_analysis(1)
        guu = self.gaussian.get_operator()
        phi1 = v[:, [0]]
        wtip1 = np.asscalar(guu @ phi1)
        lam1 = w[0]
        
        # Prepare a set of parameters used in the proceding computations.
        self.prepare_element_matrices(phi1, tau)
        self.prepare_adjoint_matrix(lam1, phi1)
        self.prepare_adjoint_gradients(lam1)
        
        # Compute the charge, stiffness, frequency, and their gradients.
        self.charge1 = self.charge()
        self.neta1 = self.charge1 / wtip1
        self.k1 = self.stiffness(phi1, wtip1)
        self.f1 = self.frequency(lam1)
        self.dneta1 = self.charge_grad(lam1, phi1, wtip1, self.charge1)
        self.dk1 = self.stiff_grad(lam1, phi1, wtip1, self.k1)
        self.df1 = self.freq_grad(lam1, phi1)
        
        
    def charge(self):
        
        charge_elem = self.fem.piezo_penalty * self.kuve_phie
        charge1 = np.sum(charge_elem)
        return charge1 
    
    
    def stiffness(self, phi, wtip):
        
        kuu = self.fem.get_stiffness_matrix()
        k1 = np.asscalar(phi.T @ kuu @ phi / wtip ** 2)
        return k1
    
    
    def frequency(self, lam):
        """Calculates the frequency given the eigenvalue `lam`.
        """
        f1 = np.sqrt(lam) / (2 * np.pi)
        return f1
    
    
    def prepare_mdof_matrix(self):
        """Each column of the operator contains the DOFs for an element.
        """
        mdofmat = np.empty((20, self.fem.mesh.n_elem), dtype=int)
        for e in self.fem.dof.dof_elements:
            mdofmat[: , e.element.index] = e.mechanical_dof
        return mdofmat
    
    
    def prepare_tdof_matrix(self):
        """Each column of the operator contains the thermal DOFs for an 
        element.
        """
        tdofmat = np.empty((4, self.fem.mesh.n_elem), dtype=int)
        for e in self.connectivity.dof.dof_elements:
            tdofmat[: , e.element.index] = e.dofs
        return tdofmat
    
    
    def prepare_element_matrices(self, phi, tau):
        
        # Each column of phie contains the mode shape over an element.
        # Each colum of taue contains the temperature field over an element.
        phie = np.squeeze(phi)[self.mdofmat]
        taue = np.squeeze(tau)[self.tdofmat]
        
        # Retrieve element matrices.
        kuve = self.fem.model.get_piezoelectric_element()
        kuue = self.fem.model.get_stiffness_element()
        muue = self.fem.model.get_mass_element()
        ktaue = self.connectivity.ke
        #ftau = self.connectivity.get_heating_matrix()
        
        # Each column is k_e * phi_e, where phi_e is the mode shape over the
        # single element and k_e is an element matrix.
        self.kuve_phie = np.squeeze(kuve.T @ phie)    # 1 x n_elem
        self.kuue_phie = np.squeeze(kuue @ phie)      # 20 x n_elem
        self.muue_phie = np.squeeze(muue @ phie)      # 20 x n_elem
        self.ktaue_taue = np.squeeze(ktaue @ taue)    # 4 x n_elem
        #self.ftaue = np.squeeze(ftau.A)[self.tdofmat] # 4 x n_elem
        
        # Row vector 1 x n_elem, each element is phi_e^T * k_e * phi_e.
        self.phi_m_phi = np.sum(np.squeeze(phie * self.muue_phie), 0)
        self.phi_k_phi = np.sum(np.squeeze(phie * self.kuue_phie), 0)
        
        
    def prepare_adjoint_matrix(self, lam, phi):
    
        # Remove the DOFs on the boundary.
        muu_free = self.fem.get_mass_matrix(free=True)
        kuu_free = self.fem.get_stiffness_matrix(free=True)
        phi_free = phi[self.fem.dof.free_dofs, :]
        
        # Compute terms of the adjoint matrix.
        a11 = kuu_free - lam * muu_free
        a12 = -2 * muu_free @ phi_free
        a21 = -phi_free.T @ muu_free
        a22 = sparse.coo_matrix((1, 1))
        
        # Stack matrices to form the adjoint matrix.
        row1 = sparse.hstack((a11, a12))
        row2 = sparse.hstack((a21, a22))
        self.adjoint = sparse.vstack((row1, row2)).tocsr()
        
        
    def prepare_adjoint_gradients(self, lam):
        """This function generates the jacobian matrices for the adjoint
        problem. There are 3 equality functions in the adjoint problem 
        denoted g_1, g_2, and g_3. These need to recomputed for each solution.
        Afterwards they are used in various gradient calculations.
        """
        # Retrieve penalization factors.
        pke = self.fem.elastic_grad 
        pme = self.fem.density_grad 
        pft = self.connectivity.heat_grad
        pkt = self.connectivity.thermal_grad 
        
        # Matric dimensions.
        n_elem = self.fem.mesh.n_elem
        n_mdof = self.fem.dof.n_mdof
        n_tdof = self.connectivity.dof.n_dof
        
        # Generate the matrix dg1_drho.
        dg1_drho_vald = (pke * self.kuue_phie - lam * pme * self.muue_phie)
        dg1_drho_row = self.mdofmat.ravel()
        dg1_drho_col = np.concatenate([np.arange(n_elem) for _ in range(20)])
        dg1_drho_val = dg1_drho_vald.ravel()
        
        dg1_drho_triplets = (dg1_drho_val, (dg1_drho_row, dg1_drho_col))
        dg1_drho_shape = (n_mdof, n_elem)
        dg1_drho = sparse.coo_matrix(dg1_drho_triplets, shape=dg1_drho_shape)
        dg1_drho_free = dg1_drho.tocsr()[self.fem.dof.free_dofs, :]
        
        # Generate the matrix dg2_drho.
        dg2_drho_free = - pme * self.phi_m_phi
        
        # Generate the matrix dg3_drho.
        dg3_drho_vald = (pkt * self.ktaue_taue - pft * self.connectivity.fe)
        dg3_drho_row = self.tdofmat.ravel()
        dg3_drho_col = np.concatenate([np.arange(n_elem) for _ in range(4)])
        dg3_drho_val = dg3_drho_vald.ravel()
        
        dg3_drho_triplets = (dg3_drho_val, (dg3_drho_row, dg3_drho_col))
        dg3_drho_shape = (n_tdof, n_elem)
        dg3_drho = sparse.coo_matrix(dg3_drho_triplets, shape=dg3_drho_shape)
        dg3_drho_free = dg3_drho.tocsr()[self.connectivity.dof.free_dofs, :]
        
        # Preserve matrices for subsequent computations.
        self.dg1_drho_free = dg1_drho_free
        self.dg2_drho_free = dg2_drho_free
        self.dg3_drho_free = dg3_drho_free
        
        
    def charge_grad(self, lam, phi, wtip, charge):

        # Retrieve data finite element matrices.
        kuv = self.fem.get_piezoelectric_matrix()
        guu = self.gaussian.get_operator()
        ktau_free = self.connectivity.get_conduction_matrix(free=True)
        ttau_free = self.connectivity.get_tau_mu_transform(free=True)
        
        # Retrieve penalization factors.
        pve = self.fem.piezo_grad 
        pte = self.fem.piezo_temp_grad
        
        # The adjoint matrix is block diagonal, compute the inverse
        # of the first block for alpha and beta.
        b1 = (kuv.T * wtip - guu * charge) / wtip ** 2
        b1 = b1[:, self.fem.dof.free_dofs]
        b2 = np.zeros((1, 1))
        bm = sparse.vstack((-b1.T, -b2.T)).tocsr()
        
        solution = sparse.linalg.spsolve(self.adjoint, bm)
        alpha = np.atleast_2d(solution[:-1]).T
        beta = solution[-1]
        
        # Compute the solution to the second block for gamma.
        dfu = (pte * self.kuve_phie) / wtip
        dft = dfu @ ttau_free
        gamma = sparse.linalg.spsolve(ktau_free, -dft.T)
        
        # Compute the derivate of the gradient with respect to the pseudo
        # densities (rho).
        dfp = pve * self.kuve_phie / wtip
        dg1p = alpha.T @ self.dg1_drho_free
        dg2p = beta * self.dg2_drho_free
        dg3p = gamma.T * self.dg3_drho_free
        
        dneta = dfp + dg1p + dg2p + dg3p
        return np.squeeze(dneta)
    
    
    def stiff_grad(self, lam, phi, wtip, k1):
        
        kuu = self.fem.get_stiffness_matrix()
        guu = self.gaussian.get_operator()
        pke = self.fem.elastic_grad
        
        b1 = 2 * phi.T @ kuu / wtip ** 2 - k1 * 2 * guu / wtip
        b1 = b1[:, self.fem.dof.free_dofs]
        b2 = np.zeros((1, 1))
        bm = sparse.vstack((-b1.T, -b2.T))
        
        solution = sparse.linalg.spsolve(self.adjoint, bm.tocsr())
        alpha = np.atleast_2d(solution[:-1]).T
        beta = solution[-1]

        dfp = (pke * self.phi_k_phi) / wtip ** 2
        dg1p = alpha.T @ self.dg1_drho_free
        dg2p = beta * self.dg2_drho_free
        dk = dfp + dg1p + dg2p

        return np.squeeze(dk)
        
    
    def freq_grad(self, lam, phi):
        """This function returns the gradient of the first modal frequency
        with respect to the pseudo-densities (rho).
        
        :param lam: The eigenvalue of the first mode.
        :param phi: The mode shape of the first mode.
        :returns: The gradient of the frequency w.r.t. the pseudo-densities.
        """
        pke = self.fem.elastic_grad
        pme = self.fem.density_grad
        df = pke * self.phi_k_phi - lam * pme * self.phi_m_phi
        df = df / (4.0 * np.pi * np.sqrt(lam))
        return np.squeeze(df)

