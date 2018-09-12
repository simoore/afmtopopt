import numpy as np
import laminate_analysis
import materials
import cantilevers
from symmetry import Symmetry
from density_filter import DensityFilter
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from projection import Projection


material = materials.PiezoMumpsMaterial()
cantilever = cantilevers.InitialCantileverFixedTip()
la = laminate_analysis.LaminateAnalysis(cantilever, material, True)
sym = Symmetry(la.fem)
density_filter = DensityFilter(la.fem, 2.0)
projection = Projection(6.0)

index = 470  # the index of the design variables to vary.
xs = np.arange(0.02, 1, 0.01)
netas = np.empty_like(xs)
ks = np.empty_like(xs)
lams = np.empty_like(xs)
dnetas = np.empty_like(xs)
dks = np.empty_like(xs)
dfs = np.empty_like(xs)

print('Number of points: %d' % len(xs))
xnew = sym.initial(la.fem.density.copy())


for i, p in enumerate(xs):
    
    xnew[index] = p
    
    x1 = sym.execute(xnew)
    x2 = density_filter.execute(x1)
    x3 = projection.execute(x2)
    dens = x3
    
    la.execute_analysis(dens)

    ddn = la.dneta1 @ projection.sensitivity(x2) @ density_filter.sensitivity() @ sym.sensitivity()
    ddk = la.dk1 @ projection.sensitivity(x2) @ density_filter.sensitivity() @ sym.sensitivity()
    ddf = la.df1 @ projection.sensitivity(x2) @ density_filter.sensitivity() @ sym.sensitivity()
    
    lams[i] = la.f1
    netas[i] = la.neta1
    ks[i] = la.k1
    dnetas[i] = ddn[index]
    dks[i] = ddk[index]
    dfs[i] = ddf[index]
    
    print('>', end='')
    

# Print characteristics.
fig, ax = plt.subplots()
ax.plot(xs, netas)
ax.set_title('Charge Characterisitic')
plt.show()

fig, ax = plt.subplots()
ax.plot(xs, ks)
ax.set_title('Stiffness Characterisitic')
plt.show()

fig, ax = plt.subplots()
ax.plot(xs, lams)
ax.set_title('Frequency Characterisitic')
plt.show()


# neta derivative
f1 = InterpolatedUnivariateSpline(xs, netas)
df1_func = f1.derivative()
df1 = df1_func(xs)

fig, ax = plt.subplots()
ax.plot(xs, df1, linewidth=3.0)
ax.plot(xs, dnetas)
ax.set_title('Charge Derivative')
plt.show()

# k derivative
f2 = InterpolatedUnivariateSpline(xs, ks)
df2_func = f2.derivative()
df2 = df2_func(xs)

fig, ax = plt.subplots()
ax.plot(xs, df2, linewidth=3.0)
ax.plot(xs, dks)
ax.set_title('Stiffness Derivative')
plt.show()

# lam derivative
f3 = InterpolatedUnivariateSpline(xs, lams)
df3_func = f3.derivative()
df3 = df3_func(xs)

fig, ax = plt.subplots()
ax.plot(xs, df3, linewidth=3.0)
ax.plot(xs, dfs)
ax.set_title('Frequency Derivative')
plt.show()
