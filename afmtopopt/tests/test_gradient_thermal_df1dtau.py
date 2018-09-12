import numpy as np
import laminate_analysis
import materials
import cantilevers
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from gaussian import Gaussian
from laminate_fem import LaminateFEM
#from connectivity import Connectivity
#import scipy.sparse as sparse

"""The mapping from tau to f1 (the objective) anf g3 (the thermal system) seem
fine. Remember that the thermal system relies soley on the free-dofs.
"""

material = materials.PiezoMumpsMaterial()
cantilever = cantilevers.InitialCantileverFixedTip()
la = laminate_analysis.LaminateAnalysis(cantilever, material, True)
fem = LaminateFEM(cantilever, material, True)
gaussian = Gaussian(fem, fem.cantilever, 0.1)

index = 1000  # the index of the pseudo-density to vary.
ps = np.arange(0.02, 1, 0.01)
netas = np.empty_like(ps)
g3s = np.empty_like(ps)
dg3_dtaus = np.empty_like(ps)
dnetas = np.empty_like(ps)
dks = np.empty_like(ps)
dfs = np.empty_like(ps)

print('Number of points: %d' % len(ps))
pnew = la.fem.density.copy()
tnew, _ = la.connectivity.thermal_analysis()
mdofmat = la.mdofmat
tdofmat = la.tdofmat

for i, p in enumerate(ps):

    tnew[index] = p

    # Update piezoelectric laminate model and perform modal analysis. 
    mu = la.connectivity.get_connectivity_penalty(tnew)
    fem.assemble(pnew, mu)
    w, v = fem.modal_analysis(1)
    guu = gaussian.get_operator()
    phi1 = v[:, [0]]
    wtip1 = np.asscalar(guu @ phi1)
    
    # Prepare a set of parameters used in the proceding computations.
    phie = np.squeeze(phi1)[mdofmat]
    kuve = fem.model.get_piezoelectric_element()
    kuve_phie = np.squeeze(kuve.T @ phie)    # 1 x n_elem
    
    # Compute the charge, stiffness, frequency, and their gradients.
    charge_elem = fem.piezo_penalty * kuve_phie
    charge1 = np.sum(charge_elem)
    neta1 = charge1 / wtip1

    ttau_free = la.connectivity.get_tau_mu_transform(free=True)
    pte = fem.piezo_temp_grad
    
    # Compute the solution to the second block for gamma.
    dfu = (pte * kuve_phie) / wtip1
    dft = dfu @ ttau_free
    
    # Since we are changing all tau, the derivatives should match.
    dft_all = np.zeros(la.connectivity.dof.n_dof)
    dft_all[la.connectivity.dof.free_dofs] = dft
    
    # g3 is the thermal system.
    ktau_free = la.connectivity.get_conduction_matrix(free=True)
    ftau_free = la.connectivity.get_heating_matrix(free=True)
    tau_free = np.expand_dims(tnew[la.connectivity.dof.free_dofs], axis=1)
    g3 = ktau_free @ tau_free - ftau_free
    #g3_all = np.zeros(la.connectivity.dof.n_dof)
    #g3_all[la.connectivity.dof.free_dofs] = g3
    dg3_dtau_all = np.zeros((len(la.connectivity.dof.free_dofs), la.connectivity.dof.n_dof))
    dg3_dtau_all[:, la.connectivity.dof.free_dofs] = ktau_free.A
    
    
    netas[i] = neta1
    g3s[i] = g3[index]
    
    dnetas[i] = dft_all[index]
    dg3_dtaus[i] = dg3_dtau_all[index, index]
    
    print('>', end='')
    

# Print characteristics.
fig, ax = plt.subplots()
ax.plot(ps, netas)
ax.set_title('Charge Characterisitic')
plt.show()

# This plot should look linear no matter the index.
fig, ax = plt.subplots()
ax.plot(ps, g3s)
ax.set_title('Thermal system')
plt.show()


# neta derivative
f1 = InterpolatedUnivariateSpline(ps, netas)
df1_func = f1.derivative()
df1 = df1_func(ps)

fig, ax = plt.subplots()
ax.plot(ps, df1, linewidth=2.0)
ax.plot(ps, dnetas)
ax.set_title('Charge Derivative with respect to tau')
plt.show()

# g3_dtauderivative
f2 = InterpolatedUnivariateSpline(ps, g3s)
df2_func = f2.derivative()
df2 = df2_func(ps)

fig, ax = plt.subplots()
ax.plot(ps, df2, linewidth=2.0)
ax.plot(ps, dg3_dtaus)
ax.set_title('dg3 dtau Derivative')
plt.show()
