import numpy as np
import laminate_analysis
import materials
import cantilevers
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from gaussian import Gaussian
from laminate_fem import LaminateFEM
from connectivity import Connectivity
#import scipy.sparse as sparse

"""The results of this analysis seem to show that the gradients are being 
computed correctly from the pseudo-densities to the gradient of the stiffness,
frequency and charge.
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
ks = np.empty_like(ps)
lams = np.empty_like(ps)
dnetas = np.empty_like(ps)
dks = np.empty_like(ps)
dfs = np.empty_like(ps)

print('Number of points: %d' % len(ps))
pnew = la.fem.density.copy()
tnew, _ = la.connectivity.thermal_analysis()
mdofmat = la.mdofmat
tdofmat = la.tdofmat

for i, p in enumerate(ps):
    
    pnew[index] = p
    #tnew[index] = p
    #la.execute_analysis(pnew)
    
    # Update the connectivity model and perform thermal analysis.
    #connectivity.assemble(pnew)
    #tau, _ = connectivity.thermal_analysis()

    # Update piezoelectric laminate model and perform modal analysis.
    mu = connectivity.get_connectivity_penalty(tnew)    
    fem.assemble(pnew, mu)
    w, v = fem.modal_analysis(1)
    guu = gaussian.get_operator()
    phi1 = v[:, [0]]
    wtip1 = np.asscalar(guu @ phi1)
    #lam1 = w[0]
    
    # Prepare a set of parameters used in the proceding computations.
    #self.prepare_element_matrices(phi1, tau)
    phie = np.squeeze(phi1)[mdofmat]
    #taue = np.squeeze(tnew)[tdofmat]
    
    # Retrieve element matrices.
    kuve = fem.model.get_piezoelectric_element()
    #kuue = fem.model.get_stiffness_element()
    #muue = fem.model.get_mass_element()
    #ktaue = connectivity.ke
    #ftau = connectivity.get_heating_matrix()
    
    # Each column is k_e * phi_e, where phi_e is the mode shape over the
    # single element and k_e is an element matrix.
    kuve_phie = np.squeeze(kuve.T @ phie)    # 1 x n_elem
    #kuue_phie = np.squeeze(kuue @ phie)      # 20 x n_elem
    #muue_phie = np.squeeze(muue @ phie)      # 20 x n_elem
    #ktaue_taue = np.squeeze(ktaue @ taue)    # 4 x n_elem
    #ftaue = np.squeeze(ftau.A)[tdofmat] # 4 x n_elem
    
    # Row vector 1 x n_elem, each element is phi_e^T * k_e * phi_e.
    #phi_m_phi = np.sum(np.squeeze(phie * muue_phie), 0)
    #phi_k_phi = np.sum(np.squeeze(phie * kuue_phie), 0)
    
    #self.prepare_adjoint_matrix(lam1, phi1)
    #self.prepare_adjoint_gradients(lam1)
    
    # Compute the charge, stiffness, frequency, and their gradients.
    charge_elem = fem.piezo_penalty * kuve_phie
    charge1 = np.sum(charge_elem)
    neta1 = charge1 / wtip1
    #self.k1 = self.stiffness(phi1, wtip1)
    #self.f1 = self.frequency(lam1)
    #self.dneta1 = self.charge_grad(lam1, phi1, wtip1, self.charge1)
    #self.dk1 = self.stiff_grad(lam1, phi1, wtip1, self.k1)
    #self.df1 = self.freq_grad(lam1, phi1)

    # Retrieve data finite element matrices.
    #kuv = fem.get_piezoelectric_matrix()
    #guu = gaussian.get_operator()
    #ktau_free = connectivity.get_conduction_matrix(free=True)
    ttau_free = connectivity.get_tau_mu_transform(free=True)
    
    # Retrieve penalization factors.
    #pve = fem.piezo_grad 
    pte = fem.piezo_temp_grad
    
    # The adjoint matrix is block diagonal, compute the inverse
    # of the first block for alpha and beta.
    #b1 = (kuv.T * wtip1 - guu * charge1) / wtip1 ** 2
    #b1 = b1[:, fem.dof.free_dofs]
    #b2 = np.zeros((1, 1))
    #bm = sparse.vstack((-b1.T, -b2.T)).tocsr()
    
    #solution = sparse.linalg.spsolve(self.adjoint, bm)
    #alpha = np.atleast_2d(solution[:-1]).T
    #beta = solution[-1]
    
    # Compute the solution to the second block for gamma.
    dfu = (pte * kuve_phie) / wtip1
    dft = dfu @ ttau_free

    #lams[i] = la.f1
    netas[i] = la.neta1
    #ks[i] = la.k1
    dnetas[i] = la.dneta1[index]
    #dks[i] = la.dk1[index]
    #dfs[i] = la.df1[index]
    
    print('>', end='')
    

# Print characteristics.
fig, ax = plt.subplots()
ax.plot(ps, netas)
ax.set_title('Charge Characterisitic')
plt.show()

#fig, ax = plt.subplots()
#ax.plot(ps, ks)
#ax.set_title('Stiffness Characterisitic')
#plt.show()
#
#fig, ax = plt.subplots()
#ax.plot(ps, lams)
#ax.set_title('Frequency Characterisitic')
#plt.show()


# neta derivative
f1 = InterpolatedUnivariateSpline(ps, netas)
df1_func = f1.derivative()
df1 = df1_func(ps)

fig, ax = plt.subplots()
ax.plot(ps, df1, linewidth=2.0)
ax.plot(ps, dnetas)
ax.set_title('Charge Derivative')
plt.show()

## k derivative
#f2 = InterpolatedUnivariateSpline(ps, ks)
#df2_func = f2.derivative()
#df2 = df2_func(ps)
#
#fig, ax = plt.subplots()
#ax.plot(ps, df2, linewidth=2.0)
#ax.plot(ps, dks)
#ax.set_title('Stiffness Derivative')
#plt.show()
#
## lam derivative
#f3 = InterpolatedUnivariateSpline(ps, lams)
#df3_func = f3.derivative()
#df3 = df3_func(ps)
#
#fig, ax = plt.subplots()
#ax.plot(ps, df3, linewidth=2.0)
#ax.plot(ps, dfs)
#ax.set_title('Frequency Derivative')
#plt.show()
