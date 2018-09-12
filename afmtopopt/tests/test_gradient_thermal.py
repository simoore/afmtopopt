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
fem = LaminateFEM(cantilever, material, True, 0.01)
connectivity = Connectivity(fem.mesh)
gaussian = Gaussian(fem, fem.cantilever, 0.1)

index = 100  # the index of the pseudo-density to vary.
ps = np.arange(0.02, 1, 0.01)
g3sums = np.empty_like(ps)
netas = np.empty_like(ps)
dg3sums = np.empty_like(ps)
dnetas = np.empty_like(ps)


print('Number of points: %d' % len(ps))
pnew = la.fem.density.copy()
mdofmat = la.mdofmat
tdofmat = la.tdofmat


for i, p in enumerate(ps):
    
    pnew[index] = p

    # Update the connectivity model and perform thermal analysis.
    connectivity.assemble(pnew)
    tau, _ = connectivity.thermal_analysis()

    # Update piezoelectric laminate model and perform modal analysis.
    mu = connectivity.get_connectivity_penalty(tau)    
    fem.assemble(pnew, mu)
    w, v = fem.modal_analysis(1)
    guu = gaussian.get_operator()
    phi1 = v[:, [0]]
    wtip1 = np.asscalar(guu @ phi1)
    lam1 = w[0]
    
    ###########################################################################
    # Prepare a set of parameters used in the proceding computations.
    # function: self.prepare_element_matrices(phi1, tau)
    
    phie = np.squeeze(phi1)[mdofmat]
    taue = np.squeeze(tau)[tdofmat]
    
    # Retrieve element matrices.
    kuve = fem.model.get_piezoelectric_element()
    kuue = fem.model.get_stiffness_element()
    muue = fem.model.get_mass_element()
    ktaue = connectivity.ke
    #ftau = connectivity.get_heating_matrix()
    
    # Each column is k_e * phi_e, where phi_e is the mode shape over the
    # single element and k_e is an element matrix.
    kuve_phie = np.squeeze(kuve.T @ phie)    # 1 x n_elem
    kuue_phie = np.squeeze(kuue @ phie)      # 20 x n_elem
    muue_phie = np.squeeze(muue @ phie)      # 20 x n_elem
    ktaue_taue = np.squeeze(ktaue @ taue)    # 4 x n_elem
    #ftaue = np.squeeze(ftau.A)[tdofmat] # 4 x n_elem
    
    # Row vector 1 x n_elem, each element is phi_e^T * k_e * phi_e.
    phi_m_phi = np.sum(np.squeeze(phie * muue_phie), 0)
    #phi_k_phi = np.sum(np.squeeze(phie * kuue_phie), 0)

    
    ###########################################################################
    # function: self.prepare_adjoint_matrix(lam1, phi1)
    
    # Remove the DOFs on the boundary.
    muu_free = fem.get_mass_matrix(free=True)
    kuu_free = fem.get_stiffness_matrix(free=True)
    phi_free = phi1[fem.dof.free_dofs, :]
    
    # Compute terms of the adjoint matrix.
    a11 = kuu_free - lam1 * muu_free
    a12 = -2 * muu_free @ phi_free
    a21 = -phi_free.T @ muu_free
    a22 = sparse.coo_matrix((1, 1))
    
    # Stack matrices to form the adjoint matrix.
    row1 = sparse.hstack((a11, a12))
    row2 = sparse.hstack((a21, a22))
    adjoint = sparse.vstack((row1, row2)).tocsr()
    
    ###########################################################################
    # function: self.prepare_adjoint_gradients(lam1)
    # Retrieve penalization factors.
    pke = fem.elastic_grad 
    pme = fem.density_grad 
    pft = connectivity.heat_grad
    pkt = connectivity.thermal_grad 
    
    # Matric dimensions.
    n_elem = fem.mesh.n_elem
    n_mdof = fem.dof.n_mdof
    n_tdof = connectivity.dof.n_dof
    
    # Generate the matrix dg1_drho.
    dg1_drho_vald = (pke * kuue_phie - lam1 * pme * muue_phie)
    dg1_drho_row = mdofmat.ravel()
    dg1_drho_col = np.concatenate([np.arange(n_elem) for _ in range(20)])
    dg1_drho_val = dg1_drho_vald.ravel()
    
    dg1_drho_triplets = (dg1_drho_val, (dg1_drho_row, dg1_drho_col))
    dg1_drho_shape = (n_mdof, n_elem)
    dg1_drho = sparse.coo_matrix(dg1_drho_triplets, shape=dg1_drho_shape)
    dg1_drho_free = dg1_drho.tocsr()[fem.dof.free_dofs, :]
    
    # Generate the matrix dg2_drho.
    dg2_drho_free = - pme * phi_m_phi
    
    # Generate the matrix dg3_drho.
    dg3_drho_vald = (pkt * ktaue_taue - pft * connectivity.fe)
    dg3_drho_row = tdofmat.ravel()
    dg3_drho_col = np.concatenate([np.arange(n_elem) for _ in range(4)])
    dg3_drho_val = dg3_drho_vald.ravel()
    
    dg3_drho_triplets = (dg3_drho_val, (dg3_drho_row, dg3_drho_col))
    dg3_drho_shape = (n_tdof, n_elem)
    dg3_drho = sparse.coo_matrix(dg3_drho_triplets, shape=dg3_drho_shape)
    dg3_drho_free = dg3_drho.tocsr()[connectivity.dof.free_dofs, :]
        
    ###########################################################################
    # Compute the charge, stiffness, frequency, and their gradients.
    charge_elem = fem.piezo_penalty * kuve_phie
    charge1 = np.sum(charge_elem)
    neta1 = charge1 / wtip1

    ###########################################################################
    # Compute derivative of dneta1 with respect to rho.
    # Retrieve data finite element matrices.
    kuv = fem.get_piezoelectric_matrix()
    guu = gaussian.get_operator()
    ktau_free = connectivity.get_conduction_matrix(free=True)
    ttau_free = connectivity.get_tau_mu_transform(free=True)
    
    # Retrieve penalization factors.
    pve = fem.piezo_grad 
    pte = fem.piezo_temp_grad
    
    # The adjoint matrix is block diagonal, compute the inverse
    # of the first block for alpha and beta.
    b1 = (kuv.T * wtip1 - guu * charge1) / wtip1 ** 2
    b1 = b1[:, fem.dof.free_dofs]
    b2 = np.zeros((1, 1))
    bm = sparse.vstack((-b1.T, -b2.T)).tocsr()
    
    solution = sparse.linalg.spsolve(adjoint, bm)
    alpha = np.atleast_2d(solution[:-1]).T
    beta = solution[-1]
    
    # Compute the solution to the second block for gamma.
    dfu = (pte * kuve_phie) / wtip1
    dft = dfu @ ttau_free
    gamma = sparse.linalg.spsolve(ktau_free, -dft.T)
    
    # Compute the derivate of the gradient with respect to the pseudo
    # densities (rho).
    dfp = pve * kuve_phie / wtip1
    dg1p = alpha.T @ dg1_drho_free
    dg2p = beta * dg2_drho_free
    dg3p = gamma.T @ dg3_drho_free
    dneta = dfp + dg1p + dg2p + dg3p

    netas[i] = neta1
    dnetas[i] = dneta[0, index]
    
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
ax.set_title('Charge Derivative')
plt.show()

