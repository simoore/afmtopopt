import numpy as np
import laminate_analysis
import materials
import cantilevers
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from gaussian import Gaussian
from laminate_fem import LaminateFEM
from connectivity import Connectivity
import scipy.sparse as sparse


"""
"""


material = materials.PiezoMumpsMaterial()
cantilever = cantilevers.InitialCantileverFixedTip()
la = laminate_analysis.LaminateAnalysis(cantilever, material, True)
fem = LaminateFEM(cantilever, material, True)
connectivity = Connectivity(fem.mesh)
gaussian = Gaussian(fem, fem.cantilever, 0.1)

index = 100  # the index of the pseudo-density to vary.
ps = np.arange(0.02, 1, 0.01)
netas = np.empty_like(ps)
dnetas = np.empty_like(ps)
dnn = np.empty_like(ps)


print('Number of points: %d' % len(ps))
pnew = la.fem.density.copy()
tau_all, tau_free = la.connectivity.thermal_analysis()
mu = la.connectivity.get_connectivity_penalty(tau_all) 
guu = gaussian.get_operator()
mdofmat = la.mdofmat
tdofmat = la.tdofmat
taue = np.squeeze(tau_all)[tdofmat]

for i, p in enumerate(ps):
    
    pnew[index] = p

    # Update the connectivity model and perform thermal analysis.
    connectivity.assemble(pnew)

    ###########################################################################
    # Prepare a set of parameters used in the proceding computations.
    # function: self.prepare_element_matrices(phi1, tau)
    
    # Retrieve element matrices.
    ktaue = connectivity.ke
    ftau = connectivity.get_heating_matrix()
    
    # Each column is k_e * phi_e, where phi_e is the mode shape over the
    # single element and k_e is an element matrix.
    ktaue_taue = np.squeeze(ktaue @ taue)    # 4 x n_elem
    #ftaue = np.squeeze(ftau.A)[tdofmat] # 4 x n_elem
    #ftaue = 
    
    
    ###########################################################################
    # function: self.prepare_adjoint_gradients(lam1)
    # Retrieve penalization factors.
    pft = connectivity.heat_grad     # correct value
    pkt = connectivity.thermal_grad  # correct value
    
    # Matric dimensions.
    n_elem = fem.mesh.n_elem
    n_tdof = connectivity.dof.n_dof
    
    # Generate the matrix dg3_drho.
    # !!!!!!!!!! ftaue is incorrect !!!!!!!!!!!!!
    dg3_drho_vald = (pkt * ktaue_taue - pft * connectivity.fe)
    dg3_drho_row = tdofmat.ravel()
    dg3_drho_col = np.concatenate([np.arange(n_elem) for _ in range(4)])
    dg3_drho_val = dg3_drho_vald.ravel()
        
    dg3_drho_triplets = (dg3_drho_val, (dg3_drho_row, dg3_drho_col))
    dg3_drho_shape = (n_tdof, n_elem)
    dg3_drho = sparse.coo_matrix(dg3_drho_triplets, shape=dg3_drho_shape)
    dg3_drho_free = dg3_drho.tocsr()[connectivity.dof.free_dofs, :]
    
    # Generate the matrix dg3_drho with a loop (much more understandable).
    #test_dg3 = []#np.zeros((1640, 1600))
    #for ii, dof in enumerate(connectivity.dof.free_dofs):
    q0, k0, eps = 1, 1e4, 1e-4
    dkdr = (k0 - eps*k0)
    dqdr = q0
    ttaue = taue[:,[index]]
    tdg3 = (dkdr * connectivity.ke) @ ttaue - dqdr * connectivity.fe
    yyyy = dkdr * ktaue_taue[:,[index]] - connectivity.fe
    #print(tdofmat[:,index]) # check that g3 doesn't contain any free dofs
    #print(connectivity.dof.free_dofs)

    
        
    ###########################################################################
    # Compute g3, the thermal system.
    
    ftau_free = connectivity.get_heating_matrix(free=True)
    ktau_free = connectivity.get_conduction_matrix(free=True)
    #if i is not 0:
    #    g3_prev = g3.copy()
    g3 = ktau_free @ tau_free - np.squeeze(ftau_free.A)

    ###########################################################################
    # Compute sum of g3 and its derivative for checking.

    netas[i] = np.sum(g3)
    dnetas[i] = np.sum(dg3_drho_free[:, index])
    dnn[i] = np.sum(tdg3)
    #print(np.count_nonzero(dg3_drho_free[:,index].A),end=' ')
    #if i is not 0:
    #    print(np.count_nonzero(g3 - g3_prev), end=' ')
    
    if i is 0:
        xxx = dg3_drho_free[:, index]
        print(np.squeeze(tdg3), end=' right\n')
        print(xxx.A[xxx.A != 0])
        print(np.squeeze(yyyy))
    
    print('>', end='')
    

# Print characteristics.
fig, ax = plt.subplots()
ax.plot(ps, netas)
ax.set_title('Charge Characterisitic')
plt.show()

# neta derivative
f1 = InterpolatedUnivariateSpline(ps, netas)
df1_func = f1.derivative()
df1 = df1_func(ps)

fig, ax = plt.subplots()
ax.plot(ps, df1, linewidth=2.0)
ax.plot(ps, dnetas)
ax.plot(ps, dnn)
ax.set_title('Charge Derivative')
plt.show()

