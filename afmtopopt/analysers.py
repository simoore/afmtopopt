import numpy as np
from matplotlib import patches, cm
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .gaussian import Gaussian
from .materials import PiezoMumpsMaterial
from .cantilevers import InitialCantileverFixedTip
from .laminate_fem import LaminateFEM
from .symmetry import Symmetry
from .density_filter import DensityFilter
from .projection import Projection


def analyze_solution():
     material = PiezoMumpsMaterial()
     cantilever = InitialCantileverFixedTip()
     fem = LaminateFEM(cantilever, material)
     analyser = CantileverAnalyser(fem)
     xs = np.load('solutions/project-b-design.npy')
     sym = Symmetry(fem)
     density_filter = DensityFilter(fem, 2.0)
     projection = Projection(6.0)
     x1 = sym.execute(xs)
     x2 = density_filter.execute(x1)
     x3 = projection.execute(x2)
     fem.assemble(x3)
     return analyser


class CantileverAnalyser(object):
    
    def __init__(self, fem):
        
        self.cantilever = fem.cantilever
        self.fem = fem
        self.img_num = 0
        plt.ioff()
        
        
    def plot_topology(self):
        for r in self.cantilever.topology:
            for e in r:
                print(int(e), end='')
            print()
            
            
    def plot_densities(self, filename=None):
        
        nelx, nely = self.fem.cantilever.topology.shape
        a = self.fem.cantilever.a
        b = self.fem.cantilever.b
        data = np.zeros((nelx, nely))
        
        x = np.linspace(0, nelx*2e6*a, nelx+1, endpoint=True)
        y = np.linspace(0, nely*2e6*b, nely+1, endpoint=True)
        xv, yv = np.meshgrid(x, y)
        
        for e, p in zip(self.fem.mesh.elements, self.fem.density_penalty):
            data[e.i, e.j] = p

        fig, ax = plt.subplots()
        ax.pcolormesh(xv, yv, data.T, cmap=cm.Greys, vmin=0, vmax=1)
        ax.set_xlim(0, nelx*2e6*a)
        ax.set_ylim(0, nely*2e6*b)
        ax.set_aspect('equal')
        plt.show()
        if filename is not None:
            #plt.tight_layout(pad=0.0)
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            
        
    def identify_modal_parameters(self, num_modes=5):
        
        w, v = self.fem.modal_analysis(num_modes)
        freq = np.sqrt(w) / 2 / np.pi
        mms = self.fem.muu
        kks = self.fem.kuu
        
        if type(self.fem).__name__ == 'LaminateFEM':
            kuv = self.fem.kuv
        
        gau = Gaussian(self.fem, self.cantilever, 0.1)
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
            if type(self.fem).__name__ == 'LaminateFEM':
                Q = -kuv.T @ phi
                eta = Q / 1e-12
                print('Charge Gain (C/m): %g' % (Q))
                print('Sensor Gain (V/m): %g' % (eta))
            print()
                    
        
    def plot_mode(self, mode_num):
        """
        :param mode_num: The mode number to plot. Note that 0 is the first
                         mode.
        """
        
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
        for n in self.fem.dof.dof_nodes:
            if n.node.void == False:
                ze[n.node.i, n.node.j] = 0
                zz[n.node.i, n.node.j] = 0
                if n.boundary == False:
                    ww = n.deflection_dof
                    zz[n.node.i, n.node.j] = v[ww, mode_num]
                vmin = min(vmin, zz[n.node.i, n.node.j])
                vmax = max(vmax, zz[n.node.i, n.node.j])
        #print('vmin, vmax: %g, %g' % (vmin, vmax))
        
        fig = figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.plot_surface(x, y, zz, rstride=1, cstride=1, linewidth=0, 
                          cmap=cm.jet, vmin=vmin, vmax=vmax)
        axis.plot_surface(x, y, ze, rstride=1, cstride=1, linewidth=0, 
                          alpha=0.3)
        #show(block=False)
        #plt.draw()
        plt.show()
#        filename = ''.join(('solutions/mode', str(self.img_num), '.png'))
#        fig.savefig(filename)
#        axis.view_init(30, 45)
#        filename = ''.join(('solutions/mode', str(self.img_num), 'r.png'))
#        fig.savefig(filename)
#        self.img_num += 1
#        plt.close(fig)
        
        
    def plot_mesh(self):
        
        fig = figure()
        subplot = fig.add_subplot(111, aspect='equal')
        for e in self.fem.mesh.elements:
            node = e.nodes[0]
            x = node.i
            y = node.j
            width, height = 1, 1
            subplot.add_patch(patches.Rectangle((x, y), width, height))
        subplot.autoscale_view(True, True, True)
        show()
        