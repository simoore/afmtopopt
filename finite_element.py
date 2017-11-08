import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import mesh


class LaminateFEM(object):
    """
    Attributes
    ----------
    self.fem_type : string
    Returns the constant 'laminate'.
    
    self.n_elem : int
    The number of elements in the finite element model.
    """
    
    def __init__(self, cantilever, material):
        
        # Public Attributes.
        self.mesh = mesh.UniformMesh(cantilever, 'laminate')
        self.fem_type = 'laminate'
        self.n_elem = self.mesh.n_elem
        self.cantilever = cantilever
        self.elements = self.mesh.get_elements()
        
        self._model = LaminateModel(material, self.mesh.a, self.mesh.b)
        
        self._muu = None
        self._kuu = None
        self._kuv = None
        self._kvv = None
        self.assemble()
        
        
    def modal_analysis(self, n_modes):
        """The return value (w) are the eigenvalues and the return value (v) 
        are the eigenvectors.
        """
        m = self._muu.tocsc()
        k = self._kuu.tocsc()
        w, v = linalg.eigsh(k, k=n_modes, M=m, sigma=0, which='LM')
        return w, v
        

    def get_mass_matrix(self):
        return self._muu
    
    
    def get_stiffness_matrix(self):
        return self._kuu
    
    
    def get_piezoelectric_matrix(self):
        return self._kuv
    
    
    def get_capacitance_matrix(self):
        return self._kvv
    
    
    def get_element_densities(self):
        return np.array([e.density for e in self.elements])
    
            
    def update_element_densities(self, densities):
        for x, e in zip(densities, self.elements):
            e.set_penalty(x)
        self.assemble()
        
        
    def assemble(self):
        """The mass, stiffness, piezoelectric, and capacitance matricies are 
        assembled in this function.
        """
        
        muue = self._model.get_mass_element()
        kuue = self._model.get_stiffness_element()
        kuve = self._model.get_piezoelectric_element()
        kvve = self._model.get_capacitance_element()
        
        nm, ne = kuve.shape
        
        k_num = nm * nm * self.n_elem
        p_num = nm * ne * self.n_elem
        c_num = ne * ne * self.n_elem
        
        k_index = list(np.ndindex(nm, nm))
        p_index = list(np.ndindex(nm, ne))
        c_index = list(np.ndindex(ne, ne))
        
        k_row = np.zeros(k_num)
        k_col = np.zeros(k_num)
        k_val = np.zeros(k_num)
        m_val = np.zeros(k_num)
        p_row = np.zeros(p_num)
        p_col = np.zeros(p_num)
        p_val = np.zeros(p_num)
        c_row = np.zeros(c_num)
        c_col = np.zeros(c_num)
        c_val = np.zeros(c_num)
        
        k_ntriplet = 0
        p_ntriplet = 0
        c_ntriplet = 0
        
        for e in self.elements:
            
            m_bound = e.get_mechanical_boundary()
            e_bound = e.get_electrical_boundary()
            m_dof = e.get_mechanical_dof()
            e_dof = e.get_electrical_dof()
            
            for ii, jj in k_index:
                if m_bound[ii] is False and m_bound[jj] is False:
                    k_row[k_ntriplet] = m_dof[ii]
                    k_col[k_ntriplet] = m_dof[jj]
                    k_val[k_ntriplet] = e.elastic_penalty * kuue[ii, jj]
                    m_val[k_ntriplet] = e.density_penalty * muue[ii, jj]
                    k_ntriplet += 1
            
            for ii, jj in p_index:
                if m_bound[ii] is False and e_bound[jj] is False:
                    p_row[p_ntriplet] = m_dof[ii]
                    p_col[p_ntriplet] = e_dof[jj]
                    p_val[p_ntriplet] = e.piezo_penalty * kuve[ii, jj]
                    p_ntriplet += 1
            
            for ii, jj in c_index:
                if e_bound[ii] is False and e_bound[jj] is False:
                    c_row[c_ntriplet] = e_dof[ii]
                    c_col[c_ntriplet] = e_dof[jj]
                    c_val[c_ntriplet] = e.cap_penalty * kvve[ii, jj]
                    c_ntriplet += 1
        
        muu_shape = (self.mesh.n_mdof, self.mesh.n_mdof)
        kuu_shape = (self.mesh.n_mdof, self.mesh.n_mdof)
        kuv_shape = (self.mesh.n_mdof, self.mesh.n_edof)
        kvv_shape = (self.mesh.n_edof, self.mesh.n_edof)
        
        self._muu = sparse.coo_matrix((m_val, (k_row, k_col)), shape=muu_shape)
        self._kuu = sparse.coo_matrix((k_val, (k_row, k_col)), shape=kuu_shape)
        self._kuv = sparse.coo_matrix((p_val, (p_row, p_col)), shape=kuv_shape)
        self._kvv = sparse.coo_matrix((c_val, (c_row, c_col)), shape=kvv_shape)
    

    def charge_grad(self, lam, phi, wtip, charge, guu):
        
        muu = self._muu
        kuu = self._kuu
        kuv = self._kuv
        muue = self._model.get_mass_element()
        kuue = self._model.get_stiffness_element()
        kuve = self._model.get_piezoelectric_element()
        
        a1 = kuu - lam * muu
        a2 = -2 * muu @ phi
        a3 = -phi.T @ muu
        a4 = sparse.coo_matrix((1, 1))
        am = sparse.vstack((sparse.hstack((a1, a2)), sparse.hstack((a3, a4))))
        
        b1 = (kuv.T * wtip - guu * charge) / wtip ** 2
        b2 = np.zeros((1, 1))
        bm = sparse.vstack((-b1.T, -b2.T))
        
        solution = sparse.linalg.spsolve(am.tocsr(), bm.tocsr())
        alpha = np.atleast_2d(solution[:-1]).T
        beta = solution[-1]
        
        dneta = np.zeros(self.n_elem)
        for i, e in enumerate(self.elements):
            phie = e.get_displacement(phi)
            alpe = e.get_displacement(alpha)
            pke = e.elastic_grad
            pme = e.density_grad
            pve = e.piezo_grad
            term_a = (pve * kuve.T @ phie) / wtip
            term_b = alpe.T @ (pke * kuue - lam * pme * muue) @ phie
            term_c = - beta * pme * phie.T @ muue @ phie
            dneta[i] = term_a + term_b + term_c
            
        return dneta
    
    
    def stiff_grad(self, lam, phi, wtip, k1, guu):
        
        muu = self._muu
        kuu = self._kuu
        muue = self._model.get_mass_element()
        kuue = self._model.get_stiffness_element()
        
        a1 = kuu - lam * muu
        a2 = -2 * muu @ phi
        a3 = -phi.T @ muu
        a4 = sparse.coo_matrix((1, 1))
        am = sparse.vstack((sparse.hstack((a1, a2)), sparse.hstack((a3, a4))))
        
        b1 = 2 * phi.T @ kuu / wtip ** 2 - k1 * 2 * guu / wtip
        b2 = np.zeros((1, 1))
        bm = sparse.vstack((-b1.T, -b2.T))
        
        solution = sparse.linalg.spsolve(am.tocsr(), bm.tocsr())
        alpha = np.atleast_2d(solution[:-1]).T
        beta = solution[-1]
        
        dk = np.empty(self.n_elem)
        for i, e in enumerate(self.elements):
            phie = e.get_displacement(phi)
            alpe = e.get_displacement(alpha)
            pke = e.elastic_grad
            pme = e.density_grad
            term_a = (pke * phie.T @ kuue @ phie) / wtip ** 2
            term_b = alpe.T @ (pke * kuue - lam * pme * muue) @ phie
            term_c = - beta * pme * phie.T @ muue @ phie
            dk[i] = term_a + term_b + term_c
            
        return dk
        
    
    def freq_grad(self, lam, phi):
    
        muue = self._model.get_mass_element()
        kuue = self._model.get_stiffness_element()
        
        df = np.empty(self.n_elem)
        for i, e in enumerate(self.elements):
            phie = e.get_displacement(phi)
            pke = e.elastic_grad
            pme = e.density_grad
            df[i] = phie.T @ (pke * kuue - lam * pme * muue) @ phie
        
        return df
        
        
            
class LaminateModel(object):
    
    _points = [[-0.577350269189626, -0.577350269189626],
               [0.577350269189626, -0.577350269189626],
               [0.577350269189626, 0.577350269189626],
               [-0.577350269189626, 0.577350269189626]]
    
    def __init__(self, material, a, b):
        self._a = a
        self._b = b
        self._jacobian = a * b
        self._material = material
        
        elements = self._generate_element_matrices()
        self._muue, self._kuue, self._kuve, self._kvve = elements
        
    
    def get_mass_element(self):
        return self._muue
    
    
    def get_stiffness_element(self):
        return self._kuue
    
    
    def get_piezoelectric_element(self):
        return self._kuve
    
    
    def get_capacitance_element(self):
        return self._kvve
    
        
    def _generate_element_matrices(self):        
        cs1, cs2, cs3, ce1, ce2, cc, cm = self._material.get_fem_parameters()
                
        muue = np.zeros((20, 20))
        kuue = np.zeros((20, 20))
        kuve = np.zeros((20, 1))
        
        # Four point integration for bending stiffness and piezoelectric effect.
        be = self._dofs_to_electric_field_matrix()
        for p in self._points:
            bs1, bs2, bs3 = self._dofs_to_strain_matrix(p)
            nu = self._dofs_to_displacement_matrix(p)
            muue += self._jacobian * nu.T @ cm @ nu
            kuue += self._jacobian * (bs1.T @ cs1 @ bs1 + bs2.T @ cs2 @ bs1)
            kuue += self._jacobian * (bs1.T @ cs2 @ bs2 + bs2.T @ cs3 @ bs2)
            kuve += self._jacobian * ((bs1.T + bs3.T) @ ce1 * be)
            kuve += self._jacobian * (bs2.T @ ce2 * be)
        
        
        # Single point integration for shear stiffness or capacitance.
        point, weight = (0, 0), 4
        _, _, bs3 = self._dofs_to_strain_matrix(point)
        kuue += weight * self._jacobian * (bs3.T @ cs1 @ bs3)
        kvve = np.array([[weight * self._jacobian * cc]])
        
        # Enforce symmetry.
        muue = 0.5 * (muue + muue.T)
        kuue = 0.5 * (kuue + kuue.T)
        return muue, kuue, kuve, kvve
    
    
    def _dofs_to_strain_matrix(self, point):
        bs1 = [None for _ in range(4)]
        bs2 = [None for _ in range(4)]
        bs3 = [None for _ in range(4)]
        for i in range(4):
            n, dndx, dndy = self._shapes(point, i)
            bs1[i] = np.array([[dndx, 0, 0, 0, 0], 
                               [0, dndy, 0, 0, 0], 
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [dndy, dndx, 0, 0, 0]])
            bs2[i] = np.array([[0, 0, 0, 0, dndx], 
                               [0, 0, 0, -dndy, 0], 
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, -dndx, dndy]])
            bs3[i] = np.array([[0, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0], 
                               [0, 0, dndy, -n, 0],
                               [0, 0, dndx, 0, n],
                               [0, 0, 0, 0, 0]])
        bs1 = np.hstack(bs1)
        bs2 = np.hstack(bs2)
        bs3 = np.hstack(bs3)
        return bs1, bs2, bs3
    
    
    def _dofs_to_displacement_matrix(self, point):
        nu = [None for _ in range(4)]
        for i in range(4):
            n, _, _ = self._shapes(point, i)
            nu[i] = np.diag((n, n, n, n, n))
        nu = np.hstack(nu)
        return nu
    
    
    def _dofs_to_electric_field_matrix(self):
        be = 1/self._material.he
        return be
    
    
    def _shapes(self, point, index):
        """The index refers to a node in the normalized element.
        index = 0 : node sw
        index = 1 : node se
        index = 2 : node ne
        index = 3 : node nw
        """
        xi, eta = point
        xi_sign = [-1, 1, 1, -1]
        eta_sign = [-1, -1, 1, 1]
        xs, es = xi_sign[index], eta_sign[index]
        n = 0.25 * (1 + xs * xi) * (1 + es * eta)
        dndx = xs * 0.25 * (1 + es * eta) / self._a
        dndy = es * 0.25 * (1 + xs * xi) / self._b
        return n, dndx, dndy
    