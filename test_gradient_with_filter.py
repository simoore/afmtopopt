import numpy as np
import laminate_analysis
import materials
import cantilevers
from symmetry import Symmetry
from density_filter import DensityFilter
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

"""Originally I was reading the density from the finite element analysis
and forgot that is heavily modified by the structural regularization. Ensure
that data is being inadvertantly modified behind the scenes.
"""

material = materials.PiezoMumpsMaterial()
cantilever = cantilevers.InitialCantileverFixedTip()
la = laminate_analysis.LaminateAnalysis(cantilever, material, 0.1, False)
sym = Symmetry(la.fem)
density_filter = DensityFilter(la.fem, 2)

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
    
    """This line was destroying the test. fem.density is updated with the 
    pseudo densities each time. It will drastically change the xnew each time.
    Instead we only want to change one element."""
    #xnew = sym.initial(la.fem.density.copy())
    xnew[index] = p
    
    x1 = sym.execute(xnew)
    x2 = density_filter.execute(x1)
    dens = x2
    
    la.execute_analysis(dens)

    ddn = la.dneta1 @ density_filter.sensitivity() @ sym.sensitivity()
    ddk = la.dk1 @ density_filter.sensitivity() @ sym.sensitivity()
    ddf = la.df1 @ density_filter.sensitivity() @ sym.sensitivity()
    
    lams[i] = la.f1
    netas[i] = 1e6 * la.neta1
    ks[i] = la.k1
    dnetas[i] = 1e6 * ddn[index]
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
