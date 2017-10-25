import numpy as np
from matplotlib import patches, cm
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gaussian import Gaussian


class CantileverAnalyser(object):
    def __init__(self, fem, cantilever):
        self.cantilever = cantilever
        self.mesh = fem.get_mesh()
        self.fem = fem
        self.img_num = 0
        plt.ioff()
        
        
    def plot_topology(self):
        for r in self.cantilever.topology:
            for e in r:
                print(int(e), end='')
            print()
            
            
    def plot_densities(self):
        
        data = np.empty((self.mesh.nelx, self.mesh.nely))
        for e in self.mesh.get_elements():
            data[e.nodes[0].i, e.nodes[0].j] = e.density_penalty
        fig, ax = plt.subplots()
        ax.pcolormesh(data.T, cmap=cm.Greys, vmin=0, vmax=1)
        plt.show()
            
        
    def identify_modal_parameters(self, xtip, ytip):
        num_modes = 4
        w, v = self.fem.modal_analysis(num_modes)
        freq = np.sqrt(w) / 2 / np.pi
        mms = self.fem.get_mass_matrix()
        kks = self.fem.get_stiffness_matrix()
        
        if self.fem.fem_type == 'laminate':
            kuv = self.fem.get_piezoelectric_matrix()
        
        gau = Gaussian(self.mesh, xtip, ytip, 0.1)
        gau = gau.get_operator()
        
        # Equivalent stiffness and mass require an equivalence in potential 
        # and kinetic energy. The mode shapes are normalised such that
        # phi.T @ mms @ phi is one.
        for i in range(num_modes):
            phi = v[:, [i]]  # the extra bracket creates column vector
            wtip = gau @ phi
            scale = 1 / wtip
            phi = phi * scale
            M = phi.T @ mms @ phi  
            K = phi.T @ kks @ phi
            
            print('--- MODE %d ---' % (i+1))
            print('Eigenvalue:        %g' % w[i])
            print('Frequency (Hz):    %g' % freq[i])
            print('Dynamic Mass:      %g' % (M))
            print('Dynamic Stiffness: %g' % (K))
            print('Tip Deflection:    %g' % wtip)
            if self.fem.fem_type == 'laminate':
                Q = -kuv.T @ phi
                eta = Q / 1e-12
                print('Charge Gain (C/m): %g' % (Q))
                print('Sensor Gain (V/m): %g' % (eta))
            print()
                    
        
    def plot_mode(self, mode_num):
        _, v = self.fem.modal_analysis(mode_num + 1)
        nelx, nely = self.cantilever.topology.shape
        xlim = nelx + 2
        ylim = nely + 2
        ze = np.full((xlim, ylim), np.nan)
        zz = np.full((xlim, ylim), np.nan)
        x = np.arange(0, xlim, 1)
        y = np.arange(0, ylim, 1)
        x, y = np.meshgrid(y, x)
        vmin, vmax = 0, 0
        for row in self.mesh.get_nodes():
            for n in row:
                if n.void == False:
                    ze[n.i, n.j] = 0
                    zz[n.i, n.j] = 0
                    if n.boundary == False:
                        ww = n.get_deflection_dof()
                        zz[n.i, n.j] = v[ww, mode_num]
                    vmin = min(vmin, zz[n.i, n.j])
                    vmax = max(vmax, zz[n.i, n.j])
        #print('vmin, vmax: %g, %g' % (vmin, vmax))
        
        fig = figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.plot_surface(x, y, zz, rstride=1, cstride=1, linewidth=0, 
                          cmap=cm.jet, vmin=vmin, vmax=vmax)
        axis.plot_surface(x, y, ze, rstride=1, cstride=1, linewidth=0, 
                          alpha=0.3)
        #show(block=False)
        #plt.draw()
        filename = ''.join(('mode', str(self.img_num), '.png'))
        fig.savefig(filename)
        axis.view_init(30, 45)
        filename = ''.join(('mode', str(self.img_num), 'r.png'))
        fig.savefig(filename)
        self.img_num += 1
        plt.close(fig)
        
        
    def plot_mesh(self):
        elements = self.mesh.get_elements()
        fig = figure()
        subplot = fig.add_subplot(111, aspect='equal')
        for e in elements:
            node = e.nodes[0]
            x = node.i
            y = node.j
            width, height = 1, 1
            subplot.add_patch(patches.Rectangle((x, y), width, height))
        subplot.autoscale_view(True, True, True)
        show()