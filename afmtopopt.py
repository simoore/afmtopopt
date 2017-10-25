import time
#import pickle
import numpy as np
import nlopt
import materials
import cantilevers
import finite_element
import projection
import density_filter

from gaussian import Gaussian
from analysers import CantileverAnalyser



def main():
    
    material = materials.PiezoMumpsMaterial()
    #cantilever = cantilevers.InitialCantileverA()
    #cantilever = cantilevers.BJNCantilever()
    cantilever = cantilevers.InitialCantileverB()
    fem = finite_element.LaminateFEM(cantilever, material)
    
    params = {}
    params['fem'] = fem
    params['k0'] = 50
    params['xtip'] = cantilever.xtip
    params['ytip_ub'] = cantilever.ytip_ub
    params['sigma'] = 0.1
    params['rmin'] = 2.0
    params['beta'] = 2.0
    
    optimizer = TopologyOptimizer(params)
    optimizer.execute()
    
    
    
    # Some testing code.
    analyser = CantileverAnalyser(fem, cantilever)
    #dens = fem.get_element_densities()
    #flt = density_filter.DensityFilter(fem.get_mesh(), 2.0)
    #dens_b = flt.execute(dens)
    #fem.update_element_densities(dens_b)
    analyser.plot_densities()
    analyser.identify_modal_parameters_with_gaussian(cantilever.xtip, optimizer.ytip)
    
    return optimizer


def perform_and_time(message, fun, *args):
    t0 = time.time()
    rtn = fun(*args)
    t1 = time.time()
    result = ' (s): %g' % (t1-t0)
    print(''.join((message, result)))
    return rtn


class TopologyOptimizer(object):
    
    def __init__(self, params):
                
        fem = params['fem']
        mesh = fem.get_mesh()
        xtip = params['xtip']
        ytip_ub = params['ytip_ub']
        sigma = params['sigma']
        beta = params['beta']
        k0 = params['k0']
        rmin = params['rmin']
        dimension = fem.n_elem + 1
        
        self._name = 'Dynamic AFM Cantilever Optimization'
        self._fem = fem
        self._density_filter = density_filter.DensityFilter(mesh, rmin)
        self._projection = projection.Projection(beta)
        self._xs_prev = None
        self._k0 = k0
        self._x0 = np.append(fem.get_element_densities(), ytip_ub)
        self._gaussian = Gaussian(fem.get_mesh(), xtip, ytip_ub, sigma)
        self._iter = 0
        
        lb = [1e-9 for _ in range(dimension)]
        ub = [1 for _ in range(dimension)]
        lb[-1] = 0
        ub[-1] = ytip_ub
        
        self._opt = nlopt.opt(nlopt.LD_MMA, dimension)
        self._opt.set_lower_bounds(lb)
        self._opt.set_upper_bounds(ub)
        self._opt.set_min_objective(self._obj_func_b)
        
        self._opt.add_inequality_constraint(self._stiffness_constraint_b, 1e-8)
        self._opt.set_xtol_rel(1e-4)
        self._opt.set_xtol_abs(1e-8)
        self._opt.set_ftol_rel(1e-4)
        self._opt.set_ftol_abs(1e-5)
        
        self.ytip = ytip_ub
        
        
    def to_console(self):
        minf = self._opt.last_optimum_value()
        print('Optimum Cost Function: %g' % minf)
        print('Optimum Solution:')
        print(self._solution)
        
        
    def execute(self):
        print(self._name)
        message = 'Timing for execution'
        self._solution = perform_and_time(message, self._opt.optimize, self._x0)
        self.ytip = self._solution[-1]

       
    def _analysis_a(self, xs, to_filter, to_project):
        
        t0 = time.time()
        dens = xs[:-1]
        ytip = xs[-1]
        
        # Structure regularization.
        if to_filter == True:
            dens1 = self._density_filter.execute(dens) 
        else:
            dens1 = dens
            
        if to_project == True:
            dens2 = self._projection.execute(dens1)
        else:
            dens2 = dens1
        
        # Finite element analysis.
        self._fem.update_element_densities(dens2)
        self._gaussian.update_ytip(ytip)
        w, v = self._fem.modal_analysis(1)
        kuu = self._fem.get_stiffness_matrix()
        kuv = self._fem.get_piezoelectric_matrix()
        guu = self._gaussian.get_operator()
        dguu = self._gaussian.get_sensitivity()
        
        phi1 = v[:, [0]]
        wtip1 = np.asscalar(guu @ phi1)
        charge1 = np.asscalar(kuv.T @ phi1)
        lam1 = np.asscalar(w)
        self._neta1 = charge1 / wtip1
        self._k1 = np.asscalar(phi1.T @ kuu @ phi1 / wtip1 ** 2)
        
        print('Cantilever stiffness is (N/m): %g' % self._k1)
        
        # Sensitivity Analysis.
        self._dneta1_dp = self._fem.charge_grad(lam1, phi1, wtip1, charge1, guu)
        self._dneta1_dy = -(dguu @ phi1) * charge1 / wtip1 ** 2
        self._dk1_dp = self._fem.stiff_grad(lam1, phi1, wtip1, self._k1, guu)
        self._dk1_dy = -2 * self._k1 * dguu @ phi1 / wtip1
        
        if to_project == True:
            dprojection = self._projection.sensitivity(dens1)
            self._dneta1_dp = self._dneta1_dp @ dprojection
            self._dk1_dp = self._dk1_dp @ dprojection
            
        if to_filter == True:
            dfilter = self._density_filter.sensitivity()
            self._dneta1_dp = self._dneta1_dp @ dfilter
            self._dk1_dp = self._dk1_dp @ dfilter
            
        t1 = time.time()
        print('Timing for analysis (s): %g' % (t1-t0))
        
        
    def _analysis_b(self, xs, to_filter, to_project):
        """This analysis is for maximising the charge sensitivity of mode 2
        while constraining the stiffness of the mode 1.
        """
        t0 = time.time()
        dens = xs[:-1]
        ytip = xs[-1]
        
        # Structure regularization.
        if to_filter == True:
            dens1 = self._density_filter.execute(dens) 
        else:
            dens1 = dens
            
        if to_project == True:
            dens2 = self._projection.execute(dens1)
        else:
            dens2 = dens1
        
        # Finite element analysis.
        self._fem.update_element_densities(dens2)
        self._gaussian.update_ytip(ytip)
        w, v = self._fem.modal_analysis(2)
        kuu = self._fem.get_stiffness_matrix()
        kuv = self._fem.get_piezoelectric_matrix()
        guu = self._gaussian.get_operator()
        dguu = self._gaussian.get_sensitivity()
        
        phi1 = v[:, [0]]
        phi2 = v[:, [1]]
        wtip1 = np.asscalar(guu @ phi1)
        wtip2 = np.asscalar(guu @ phi2)
        charge2 = np.asscalar(kuv.T @ phi2)
        lam1 = w[0]
        lam2 = w[1]
        self._neta2 = charge2 / wtip2
        self._k1 = np.asscalar(phi1.T @ kuu @ phi1 / wtip1 ** 2)
        
        print('Charge sensitivity of mode 2 (C/m): %g' % charge2)
        print('Stiffness of mode 1 (N/m): %g' % self._k1)
        
        # Sensitivity Analysis.
        self._dneta2_dp = self._fem.charge_grad(lam2, phi2, wtip2, charge2, guu)
        self._dneta2_dy = -(dguu @ phi2) * charge2 / wtip2 ** 2
        self._dk1_dp = self._fem.stiff_grad(lam1, phi1, wtip1, self._k1, guu)
        self._dk1_dy = -2 * self._k1 * dguu @ phi1 / wtip1
        
        if to_project == True:
            dprojection = self._projection.sensitivity(dens1)
            self._dneta2_dp = self._dneta2_dp @ dprojection
            self._dk1_dp = self._dk1_dp @ dprojection
            
        if to_filter == True:
            dfilter = self._density_filter.sensitivity()
            self._dneta2_dp = self._dneta2_dp @ dfilter
            self._dk1_dp = self._dk1_dp @ dfilter
            
        t1 = time.time()
        print('Timing for analysis (s): %g' % (t1-t0))
        
                
    def _obj_func_a(self, xs, grad):
        
        self._iter += 1
        print('Iteration %d' % self._iter)
        
        if np.array_equal(xs, self._xs_prev) == False:
            self._xs_prev = xs.copy()
            self._analysis_a(xs, True, False)
        
        scale = 1e6
        if grad.size > 0:
            grad[:-1] = scale * self._dneta1_dp
            grad[-1] = scale * self._dneta1_dy
        return scale * self._neta1
    
    
    def _obj_func_b(self, xs, grad):
        
        self._iter += 1
        print('Iteration %d' % self._iter)
        
        if np.array_equal(xs, self._xs_prev) == False:
            self._xs_prev = xs.copy()
            self._analysis_b(xs, True, False)
        
        scale = 1e6
        if grad.size > 0:
            grad[:-1] = scale * self._dneta2_dp
            grad[-1] = scale * self._dneta2_dy
        return scale * self._neta2
    
    
    def _stiffness_constraint_a(self, xs, grad):
        
        if np.array_equal(xs, self._xs_prev) == False:
            self._xs_prev = xs.copy()
            self._analysis_a(xs, True, False)
        if grad.size > 0:
            grad[:-1] = self._dk1_dp  
            grad[-1] = self._dk1_dy  
        return self._k1 - self._k0
    
    
    def _stiffness_constraint_b(self, xs, grad):
        
        if np.array_equal(xs, self._xs_prev) == False:
            self._xs_prev = xs.copy()
            self._analysis_b(xs, True, False) 
        if grad.size > 0:
            grad[:-1] = self._dk1_dp  
            grad[-1] = self._dk1_dy  
        return self._k1 - self._k0
    
    
#    def save_data(self):
#        with open(self.filename, 'wb') as fp:
#            pickle.dump(self.data, fp)
#            
#            
#    def load_data(self):
#        with open(self.filename, 'rb') as fp:
#            self.data = pickle.load(fp)
        
        
if __name__ == '__main__':
    opt = main()