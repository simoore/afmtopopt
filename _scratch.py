from materials import PiezoMumpsMaterial
from cantilevers import BJNCantilever
from finite_element import LaminateFEM
import numpy as np
import time
import scipy.sparse as sparse
import scipy

material = PiezoMumpsMaterial()
cantilever = BJNCantilever()
fem = LaminateFEM(cantilever, material)
lam, phi = fem.modal_analysis(1)

#import warnings
#from scipy.sparse import SparseEfficiencyWarning
#warnings.simplefilter('ignore', SparseEfficiencyWarning)

# This runs faster but throws a warning when performing the row slice.
# No use converting it to 
t0 = time.time()
LHSB = fem._kuu - lam[0] * fem._muu
LHSB[0,:] = np.squeeze(2 * phi.T @ fem._muu)
LHSB = LHSB.tocsc()
#INV = linalg.inv()
t1 = time.time()
print(t1-t0)


# This took 64 seconds, all due to the inv matrix instruction.
t0 = time.time()
LHSB = fem._kuu - lam[0] * fem._muu
LHSB[0,:] = np.squeeze(2 * phi.T @ fem._muu)
#LHSB = LHSB.tocsc()
#INV = np.linalg.inv(LHSB.todense())
t1 = time.time()
print(t1-t0)

t0 = time.time()

e = fem._elements[0]
phie = e.get_displacement(phi)
pke = e.elastic_grad
pme = e.density_grad
kuue = fem._model.get_stiffness_element()
muue = fem._model.get_mass_element()
term_a = (pke * kuue - lam * pme * muue) @ phie
dlam = phie.T @ term_a
#LHSB.todense()

b_mat = fem._muu @ phi
dof = e.get_mechanical_dof()
boundary = e.get_mechanical_boundary()
for i in range(len(dof)):
    if boundary[i] == False:
        b_mat[dof[i]] += term_a[i]
b_mat[0] = pme * phie.T @ muue @ phie
#dphi = scipy.linalg.solve(LHSB.todense(), b_mat)
dphi = sparse.linalg.gmres(LHSB, b_mat)
t1 = time.time()
print(t1-t0)

#t0 = time.time()
#LHS = fem._kuu - lam[0] * fem._muu
#LHS = LHS.tolil()
#LHS[0,:] = np.ones(12400)
#LHS = LHS.tocsr()
#t1 = time.time()
#print(t1-t0)
#
#
#t2 = time.time()
#a_row = np.zeros(12400)
#a_col = np.arange(12400)
#a_val = np.ones(12400)
#rows = np.concatenate((fem._kuu.row, fem._muu.row, a_row))
#cols = np.concatenate((fem._kuu.row, fem._muu.col, a_col))
#vals = np.concatenate((fem._kuu.data, -lam[0]*fem._muu.data, a_val))
#a_mat = sparse.coo_matrix((vals, (rows, cols)), shape=fem._kuu.shape).tocsr()
#t3 = time.time()
#print(t3-t2)
#
#t2 = time.time()
#a_row = np.zeros(12400)
#a_col = np.arange(12400)
#a_val = np.ones(12400)
#a_mat = sparse.coo_matrix((a_val, (a_row, a_col)), shape=fem._kuu.shape).tocsr()
#LHSA = fem._kuu - lam[0] * fem._muu
#try:
#    LHSA[0,:] = 0
#except:
#    pass
#t3 = time.time()
#print(t3-t2)
#
#LHSC = fem._kuu - lam[0] * fem._muu
#print(LHSC[0,0])
#print(LHSA[0,0])

#dlam, dphi = fem.modal_sensitivity(lam, phi)








#    def _assemble_mechanical_matrices(self):
#        
#        muue = self._elements.get_mass_element()
#        kuue = self._elements.get_stiffness_element()
#        nr, nc = muue.shape
#        
#        num = nr * nc * len(self.mesh.elements)
#        row = np.zeros(num)
#        col = np.zeros(num) 
#        kk = np.zeros(num)
#        mm = np.zeros(num) 
#        ntriplet = 0
#        
#        for e in self.mesh.elements:
#            boundary = e.get_boundary()
#            dof = e.get_mechanical_dof()
#            for ii in range(nr):
#                for jj in range(nc):
#                    if boundary[ii] is False and boundary[jj] is False:
#                        row[ntriplet] = dof[ii]
#                        col[ntriplet] = dof[jj]
#                        mm[ntriplet] = e.density_penalty * muue[ii, jj]
#                        kk[ntriplet] = e.elastic_penalty * kuue[ii, jj]
#                        ntriplet += 1
#        
#        shape = (self.mesh.n_mdof, self.mesh.n_mdof)
#        kuu = sparse.coo_matrix((kk, (row, col)), shape=shape)
#        muu = sparse.coo_matrix((mm, (row, col)), shape=shape)
#        return muu, kuu
    
    
#    def _assemble_piezoelectric_matrix(self):
#        
#        # Element piezoelectric matrix and its dimensions.
#        kuve = self._elements.get_piezoelectric_element()
#        nr, nc = kuve.shape
#        
#        num = nr * nc * len(self.mesh.elements)
#        row = np.zeros(num)
#        col = np.zeros(num)
#        uv = np.zeros(num)
#        ntriplet = 0
#        
#        for e in self.mesh.elements:
#            boundary = e.get_boundary()
#            dof_m = e.get_mechanical_dof()
#            dof_e = e.get_electrical_dof()
#            for ii in range(nr):
#                for jj in range(nc):
#                    if boundary[ii] is False:
#                        row[ntriplet] = dof_m[ii]
#                        col[ntriplet] = dof_e[jj]
#                        uv[ntriplet] = e.elastic_penalty * kuve[ii, jj]
#                        ntriplet += 1
#        
#        shape = (self.mesh.n_mdof, self.mesh.n_edof)
#        kuv = sparse.coo_matrix((uv, (row, col)), shape=shape).tocsc()
#        return kuv
      
#        kvve = self._elements.get_capacitance_element()
#        n_edof = self.mesh.n_edof
#        vv = np.zeros(n_edof)
#        for e in self.mesh.elements:
#            edof = e.get_electrical_dof()
#            vv[edof] += kvve
#        shape = (n_edof, n_edof)
#        kvv = sparse.diags(vv, shape=shape)
#        return kvv



#    def generate_stiffness_sensitivity(self, n_mdof, pe):
#        """The stiffness sensitivity is element wise.
#        """
#        nr, nc = self._kuue.shape
#        num = nr * nc
#        row = np.zeros(num)
#        col = np.zeros(num)
#        dke = np.zeros(num)
#        scale = pe * self._density ** (pe - 1)
#        ntriplet = 0
#        
#        boundary = self._mesh_element.get_boundary()
#        dof = self._mesh_element.get_mechanical_dof()
#        for ii in range(nr):
#            for jj in range(nc):
#                if boundary[ii] is False:
#                    row[ntriplet] = dof[ii]
#                    col[ntriplet] = dof[jj]
#                    dke[ntriplet] = scale * self._kuue
#                    ntriplet += 1
#        
#        shape = (n_mdof, n_mdof)
#        mat = sparse.coo_matrix((dke, (row, col)), shape=shape)
#        return mat 
#    
#    
#    def generate_mass_sensitivity(self, n_mdof, pe):
#        """The stiffness sensitivity is element wise.
#        """
#        nr, nc = self._muue.shape
#        num = nr * nc
#        row = np.zeros(num)
#        col = np.zeros(num)
#        dke = np.zeros(num)
#        scale = pe * self._density ** (pe - 1)
#        ntriplet = 0
#        
#        boundary = self._mesh_element.get_boundary()
#        dof = self._mesh_element.get_mechanical_dof()
#        for ii in range(nr):
#            for jj in range(nc):
#                if boundary[ii] is False:
#                    row[ntriplet] = dof[ii]
#                    col[ntriplet] = dof[jj]
#                    dke[ntriplet] = scale * self._muue
#                    ntriplet += 1
#        
#        shape = (n_mdof, n_mdof)
#        mat = sparse.coo_matrix((dke, (row, col)), shape=shape)
#        return mat 