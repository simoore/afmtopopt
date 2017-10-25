import time
import pickle
import numpy as np
import nlopt
import materials
import cantilevers
import finite_element
import projection
import density_filter
import gaussian
import analysers


def main():
    
    material = materials.PiezoMumpsMaterial()
    cantilever = InitialCantileverFixedTip()
    fem = finite_element.LaminateFEM(cantilever, material)
    
    params = {}
    params['fem'] = fem
    params['k0'] = 50.0
    params['xtip'] = cantilever.xtip
    params['ytip'] = cantilever.ytip
    params['sigma'] = 0.1
    params['rmin'] = 2.0
    params['beta'] = 2.0
    params['f0'] = 20.0e3
    
    analyser = analysers.CantileverAnalyser(fem, cantilever)
    #analyser.plot_densities()
    analyser.identify_modal_parameters(cantilever.xtip, cantilever.ytip)
    
    optimizer = TopologyOptimizer(params)
    optimizer.execute()
    
    print('\n--- Solution Analysis ---')
    analyser.plot_densities()
    analyser.identify_modal_parameters(cantilever.xtip, cantilever.ytip)
    
    return optimizer


class InitialCantileverFixedTip(cantilevers.Cantilever):
    
    def __init__(self):
        
        a = 10.0e-6
        b = 10.0e-6
        nelx = 40
        nely = 40
        xtip = 1e6 * (a * nelx)
        ytip = 1e6 * (2 * nely * b - b)
        
        topology = np.ones((nelx, nely))
        
        top = 1e-4 * np.ones((25, nely))
        mid = np.ones((30, nely))
        bot = 1e-4 * np.ones((25, nely))
        densities = np.vstack((top, mid, bot))
        
        super().__init__(topology, a, b, xtip, ytip, densities)
        
        print('--- Initial Design Space Fixed Tip ---')
        print('Each element is %g x %g um' % (2*a*1e6, 2*b*1e6))
        print('The design area is %g x %g um' % (2*a*nelx*1e6, 2*b*nely*1e6))
        print('(xtip, ytip) = (%g, %g) um\n' % (self.xtip, self.ytip))
        

class TopologyOptimizer(object):
    
    def __init__(self, params):
                
        fem = params['fem']
        xtip = params['xtip']
        ytip = params['ytip']
        sigma = params['sigma']
        beta = params['beta']
        k0 = params['k0']
        f0 = params['f0']
        rmin = params['rmin']
        
        mesh = fem.get_mesh()
        dimension = fem.n_elem
        
        self.name = 'Dynamic AFM Cantilever Optimization'
        self.fem = fem
        self.density_filter = density_filter.DensityFilter(mesh, rmin)
        self.projection = projection.Projection(beta)
        self.k0 = k0
        self.f0 = f0
        self.x0 = fem.get_element_densities()
        #self.x0 = self.load_solution()
        self.gaussian = gaussian.Gaussian(mesh, xtip, ytip, sigma)
        
        self.analyser = analysers.CantileverAnalyser(fem, fem.cantilever)
        
        # Attributes not set in contructor.
        self.xs_prev = None
        self.iter = 0
        self.neta1 = 0.0
        self.k1 = 0.0
        self.f1 = 0.0
        self.solution = None
        self.dneta_dp = None
        self.df1_dp = None
        self.dk1_dp = None
        
        lb = 1e-4 * np.ones(dimension) 
        ub = np.ones(dimension)
        
        self.opt = nlopt.opt(nlopt.LD_MMA, dimension)
        self.opt.set_lower_bounds(lb)
        self.opt.set_upper_bounds(ub)
        self.opt.set_min_objective(self.obj_func)
        
        self.opt.add_inequality_constraint(self.stiffness_constraint, 1e-8)
        self.opt.add_inequality_constraint(self.frequency_constraint, 1e-8)
        self.opt.set_xtol_rel(1e-10)
        self.opt.set_xtol_abs(1e-10)
        self.opt.set_ftol_rel(1e-10)
        self.opt.set_ftol_abs(1e-10)
        self.opt.set_maxeval(20000)
        
        
    def to_console(self):
        
        minf = self.opt.last_optimum_value()
        print('Optimum Cost Function: %g' % minf)
        print('Optimum Solution:')
        print(self.solution)
        
        
    def execute(self):
        
        messages = {3: 'Optimization stopped because ftol_rel or ftol_abs.',
                    4: 'Optimization stopped because xtol_rel or xtol_abs.',
                    5: 'Optimization stopped because maxeval.'}
        
        print(self.name)
        t0 = time.time()
        self.solution = self.opt.optimize(self.x0)
        self.save_solution()
        t1 = time.time()
        print('Timing for execution (s): %g\n' % (t1-t0))
        
        result = self.opt.last_optimize_result()
        default = ''.join(('No defined message for number ', str(result), '.'))
        print(messages.get(result, default))

       
    def analysis(self, xs, to_filter, to_project):
        
        t0 = time.time()
        
        # Structure regularization.
        if to_filter == True:
            dens1 = self.density_filter.execute(xs) 
        else:
            dens1 = xs
            
        if to_project == True:
            dens2 = self.projection.execute(dens1)
        else:
            dens2 = dens1
        
        # Finite element analysis.
        self.fem.update_element_densities(dens2)
        w, v = self.fem.modal_analysis(1)
        kuu = self.fem.get_stiffness_matrix()
        kuv = self.fem.get_piezoelectric_matrix()
        guu = self.gaussian.get_operator()
        
        phi1 = v[:, [0]]
        wtip1 = np.asscalar(guu @ phi1)
        charge1 = np.asscalar(kuv.T @ phi1)
        lam1 = w[0]
        self.neta1 = charge1 / wtip1
        self.k1 = np.asscalar(phi1.T @ kuu @ phi1 / wtip1 ** 2)
        self.f1 = np.sqrt(lam1) / (2 * np.pi)
        
        print('Cantilever neg. charge sens is (C/m): %g' % self.neta1)
        print('Cantilever stiffness is (N/m):        %g' % self.k1)
        print('Cantilever frequency is (Hz):         %g' % self.f1)
        
        # Sensitivity Analysis.
        self.dneta1_dp = self.fem.charge_grad(lam1, phi1, wtip1, charge1, guu)
        self.dk1_dp = self.fem.stiff_grad(lam1, phi1, wtip1, self.k1, guu)
        self.df1_dp = self.fem.freq_grad(lam1, phi1)
        
        if to_project == True:
            dprojection = self.projection.sensitivity(dens1)
            self.dneta1_dp = self.dneta1_dp @ dprojection
            self.dk1_dp = self.dk1_dp @ dprojection
            self.df1_dp = self.df1_dp @ dprojection
            
        if to_filter == True:
            dfilter = self.density_filter.sensitivity()
            self.dneta1_dp = self.dneta1_dp @ dfilter
            self.dk1_dp = self.dk1_dp @ dfilter
            self.df1_dp = self.df1_dp @ dfilter
            
        t1 = time.time()
        print('Timing for analysis (s): %g\n' % (t1-t0))
        
        
    def obj_func(self, xs, grad):
        
        self.iter += 1
        print('>>> Iteration %d' % self.iter)
        
        if np.array_equal(xs, self.xs_prev) == False:
            self.xs_prev = xs.copy()
            self.analysis(xs, True, False)
        #self.analyser.plot_mode(0)
        #input("Press Enter to continue...")
        scale = 1e6
        if grad.size > 0:
            grad[:] = scale * self.dneta1_dp
        return scale * self.neta1
    
        
    def stiffness_constraint(self, xs, grad):
        
        if np.array_equal(xs, self.xs_prev) == False:
            self.xs_prev = xs.copy()
            self.analysis(xs, True, False)
            
        if grad.size > 0:
            grad[:] = self.dk1_dp  
        return self.k1 - self.k0
    
    
    def frequency_constraint(self, xs, grad):
        
        if np.array_equal(xs, self.xs_prev) == False:
            self.xs_prev = xs.copy()
            self.analysis(xs, True, False)
        
        if grad.size > 0:
            grad[:] = -self.df1_dp
        return self.f0 - self.f1
    
    
    def save_solution(self):
        with open('solution-a.txt', 'wb') as fp:
            pickle.dump(self.solution, fp)
            
    
    def load_solution(self):
        with open('solution-a.txt', 'rb') as fp:
            solution = pickle.load(fp)
        return solution
            
    
    def save_data(filename, data):
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp)
            
            
    def load_data(filename):
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
        return data
        
        
if __name__ == '__main__':
    opt = main()