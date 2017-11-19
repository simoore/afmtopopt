import numpy as np
import laminate_analysis
import materials
import cantilevers
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

"""The results of this analysis seem to show that the gradients are being 
computed correctly from the pseudo-densities to the gradient of the stiffness,
frequency and charge.
"""

material = materials.PiezoMumpsMaterial()
cantilever = cantilevers.InitialCantileverFixedTip()
la = laminate_analysis.LaminateAnalysis(cantilever, material, 0.1, False)

index = 1000  # the index of the pseudo-density to vary.
ps = np.arange(0.02, 1, 0.01)
netas = np.empty_like(ps)
ks = np.empty_like(ps)
lams = np.empty_like(ps)
dnetas = np.empty_like(ps)
dks = np.empty_like(ps)
dfs = np.empty_like(ps)

print('Number of points: %d' % len(ps))

for i, p in enumerate(ps):
    
    pnew = la.fem.density.copy()
    pnew[index] = p
    la.execute_analysis(pnew)

    lams[i] = la.f1
    netas[i] = 1e6 * la.neta1
    ks[i] = la.k1
    dnetas[i] = 1e6 * la.dneta1[index]
    dks[i] = la.dk1[index]
    dfs[i] = la.df1[index]
    
    print('i', end='')
    

# Print characteristics.
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
ax.plot(ps, df1, linewidth=2.0)
ax.plot(ps, dnetas)
ax.set_title('Charge Derivative')
plt.show()

# k derivative
f2 = InterpolatedUnivariateSpline(ps, ks)
df2_func = f2.derivative()
df2 = df2_func(ps)

fig, ax = plt.subplots()
ax.plot(ps, df2, linewidth=2.0)
ax.plot(ps, dks)
ax.set_title('Stiffness Derivative')
plt.show()

# lam derivative
f3 = InterpolatedUnivariateSpline(ps, lams)
df3_func = f3.derivative()
df3 = df3_func(ps)

fig, ax = plt.subplots()
ax.plot(ps, df3, linewidth=2.0)
ax.plot(ps, dfs)
ax.set_title('Frequency Derivative')
plt.show()
