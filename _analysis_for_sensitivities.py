import numpy as np
import finite_element
import materials
import cantilevers
import gaussian
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

class OneElementCantilever(cantilevers.Cantilever):
    
    def __init__(self):
        
        a = 4e-6
        b = 10e-6
        topology = np.ones((1, 1))
        xtip = 5
        ytip = 8
        super().__init__(topology, a, b, xtip, ytip, topology)
        
        
class SixElementCantilever(cantilevers.Cantilever):
    
    def __init__(self):
        
        a = 4e-6
        b = 10e-6
        topology = np.ones((2, 3))
        xtip = 8
        ytip = 55
        super().__init__(topology, a, b, xtip, ytip, topology)


material = materials.PiezoMumpsMaterial()
cantilever = OneElementCantilever()
fem = finite_element.LaminateFEM(cantilever, material)

#print('-- Finite Element Matrices --')
#print(fem.get_mass_matrix().todense())
#print(fem.get_stiffness_matrix().todense())
#print(fem.get_piezoelectric_matrix().todense())
#print(fem.get_capacitance_matrix().todense())

# The single element is longer, rather than short to avoid the torsional mode
# coming in as the first mode.

ps = np.arange(0.02, 1, 0.01)
netas = np.empty_like(ps)
ks = np.empty_like(ps)
lams = np.empty_like(ps)
dnetas = np.empty_like(ps)
dks = np.empty_like(ps)
dfs = np.empty_like(ps)
print(len(ps))


for i, p in enumerate(ps):
    
    fem.update_element_densities([p])
    w, v = fem.modal_analysis(1)

    kuu = fem.get_stiffness_matrix()
    kuv = fem.get_piezoelectric_matrix()
    gau = gaussian.Gaussian(fem.get_mesh(), 4, 18, 0.05)
    guu = gau.get_operator()

    phi1 = v[:, [0]]
    wtip1 = np.asscalar(guu @ phi1)
    charge1 = np.asscalar(kuv.T @ phi1)
    lams[i] = np.asscalar(w)
    netas[i] = 1e6 * charge1 / wtip1
    ks[i] = np.asscalar(phi1.T @ kuu @ phi1 / wtip1 ** 2)
    dnetas[i] = 1e6 * np.asscalar(fem.charge_grad(lams[i], phi1, wtip1, charge1, guu))
    dks[i] = np.asscalar(fem.stiff_grad(lams[i], phi1, wtip1, ks[i], guu))
    dfs[i] = np.asscalar(fem.freq_grad(lams[i], phi1))

    
fig, ax = plt.subplots()
ax.plot(ps, netas)
ax.set_title('Charge Characterisitic')
plt.show()

fig, ax = plt.subplots()
ax.plot(ps, ks)
ax.set_title('Stiffness Characterisitic')
plt.show()

fig, ax = plt.subplots()
ax.plot(ps, lams)
ax.set_title('Frequency Characterisitic')
plt.show()


# neta derivative
f1 = InterpolatedUnivariateSpline(ps, netas)
df1_func = f1.derivative()
df1 = df1_func(ps)

fig, ax = plt.subplots()
ax.plot(ps, df1)
ax.plot(ps, dnetas)
ax.set_title('Charge Derivative')
plt.show()

# k derivative
f2 = InterpolatedUnivariateSpline(ps, ks)
df2_func = f2.derivative()
df2 = df2_func(ps)

fig, ax = plt.subplots()
ax.plot(ps, df2)
ax.plot(ps, dks)
ax.set_title('Stiffness Derivative')
plt.show()

# lam derivative
f3 = InterpolatedUnivariateSpline(ps, lams)
df3_func = f3.derivative()
df3 = df3_func(ps)

fig, ax = plt.subplots()
ax.plot(ps, df3)
ax.plot(ps, dfs)
ax.set_title('Frequency Derivative')
plt.show()
