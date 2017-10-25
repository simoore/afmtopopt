import numpy as np
import scipy.sparse as sparse
from mesh import UniformMesh


class PlateFEM(object):
    
    def __init__(self, cantilever, material):
        self.cantilever = cantilever
        self.mesh = UniformMesh(cantilever, 'plate')
        self.material = material
        self.num_elem = self.mesh.n_elem
        self.ke, self.me = self.normalised_elements()
        
        self.fem_type = 'plate'
        self.muu = None
        self.kuu = None
        self.assemble()
        
        
    def get_mesh(self):
        return self.mesh
    
        
    def get_mass_matrix(self):
        return self.muu
    
    
    def get_stiffness_matrix(self):
        return self.kuu
    
    
    def modal_analysis(self, n_modes):
        kks = self.kuu
        mms = self.muu
        w, v = sparse.linalg.eigsh(kks, k=n_modes, M=mms, sigma=0, which='LM')
        return w, v
    
    
    def assemble(self):
        ntriplet = 0
        num = 144 * len(self.mesh.get_elements())
        row, col = np.zeros(num), np.zeros(num)
        kk, mm = np.zeros(num), np.zeros(num)
        for e in self.mesh.get_elements():
            boundary = e.get_mechanical_boundary()
            dof = e.get_mechanical_dof()
            for ii in range(12):
                for jj in range(12):
                    if boundary[ii] is False and boundary[jj] is False:
                        row[ntriplet] = dof[ii]
                        col[ntriplet] = dof[jj]
                        mm[ntriplet] = self.me[ii, jj]
                        kk[ntriplet] = self.ke[ii, jj]
                        ntriplet += 1
                 
        shape = (self.mesh.n_mdof, self.mesh.n_mdof)
        kks = sparse.coo_matrix((kk, (row, col)), shape=shape).tocsc()
        mms = sparse.coo_matrix((mm, (row, col)), shape=shape).tocsc()
        self.kuu = kks
        self.muu = mms


    def normalised_elements(self):
        CI, CO, CM, H, Kappa = self.material.get_finite_element_parameters()
        A, B = self.mesh.get_finite_element_parameters()
        
        ke = np.zeros((12, 12))
        me = np.zeros((12, 12))
        
        points = [[-0.577350269189626, -0.577350269189626],
                  [0.577350269189626, -0.577350269189626],
                  [0.577350269189626, 0.577350269189626],
                  [-0.577350269189626, 0.577350269189626]]
        
        # Numerical integration to evaluate the element mass and stiffness.
        for p in points:
            BI = [None for _ in range(4)]
            BM = [None for _ in range(4)]
            for i in range(4):
                n, dndx, dndy = self.shapes(p, A, B, i)
                BI[i] = np.array([[0, 0, -dndx], [0, dndy, 0], [0, dndx, -dndy]])
                BM[i] = np.diag((n, n, n))
            BI = np.hstack(BI)
            BM = np.hstack(BM)
            ke += A*B * H**3/12 * BI.T @ CI @ BI
            me += A*B * BM.T @ CM @ BM
        
        # Single point integration for shear stiffness.
        point, weight = (0, 0), 4
        BO = [None for _ in range(4)]
        for i in range(4):
            n, dndx, dndy = self.shapes(point, A, B, i)
            BO[i] = np.array([[dndx, 0, n], [dndy, -n, 0]])
        BO = np.hstack(BO)
        ke += weight * A*B * Kappa*H * BO.T @ CO @ BO
        
        # Enforce symmetry.
        me = 0.5 * (me + me.T)
        ke = 0.5 * (ke + ke.T)
        return ke, me

    
    def shapes(self, point, A, B, index):
        '''The index refers to a node in the normalized element.
        index = 0 : node sw
        index = 1 : node se
        index = 2 : node ne
        index = 3 : node nw
        '''
        xi, eta = point
        xi_sign = [-1, 1, 1, -1]
        eta_sign = [-1, -1, 1, 1]
        xs, es = xi_sign[index], eta_sign[index]
        n = 0.25 * (1 + xs * xi) * (1 + es * eta)
        dndx = xs * 0.25 * (1 + es * eta) / A
        dndy = es * 0.25 * (1 + xs * xi) / B
        return n, dndx, dndy
    