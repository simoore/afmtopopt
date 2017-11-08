import time
import numpy as np
import ipopt
import logging
from ruamel.yaml import YAML
import pprint
import os


import materials
import cantilevers
import finite_element
import projection
import density_filter
import gaussian
import analysers
import symmetry


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    params = load_parameters('solutions/solution-c.yaml')
    print('-- Parameters --\n')
    pprint.pprint(params)
    print()
    
    optimizer = TopologyOptimizer(params)
    optimizer.cantilever.to_console()
    
    print('\n--- Initial Analysis ---')
    optimizer.analyser.plot_densities()
    optimizer.analyser.identify_modal_parameters()
    
    optimizer.execute()
    return optimizer


def load_parameters(filename):
    
    with open(filename, 'r') as f:
        yaml = YAML(typ='safe')
        params = yaml.load(f)
        params['tag'] = os.path.splitext(os.path.basename(f.name))[0]
        params['dir'] = os.path.dirname(f.name)
        return params


class TopologyOptimizer(object):
    
    def __init__(self, params):
                
        sigma = params['sigma']
        beta = params['beta']
        k0 = params['k0']
        f0 = params['f0']
        rmin = params['rmin']
        max_iter = params['max_iter']
        self.tag = params['tag']
        self.debug = params['debug']
        self.to_filter = params['to_filter']
        self.to_project = params['to_project']
        self.cantilever_key = params['cantilever']
        self.dir = params['dir']
        
        self.material = materials.PiezoMumpsMaterial()
        self.cantilever = self.select_cantilever()
        self.fem = finite_element.LaminateFEM(self.cantilever, self.material)
        self.analyser = analysers.CantileverAnalyser(self.fem)
        
        self.sym = symmetry.Symmetry(self.fem)
        self.density_filter = density_filter.DensityFilter(self.fem, rmin)
        self.projection = projection.Projection(beta)
        self.gaussian = gaussian.Gaussian(self.fem, self.cantilever, sigma)
        self.x0 = self.sym.initial(self.fem.get_element_densities())
        #self.x0 = self.load_solution()
        
        # Attributes not set in contructor.
        self.xs_prev = None
        self.neta1 = 0.0
        self.k1 = 0.0
        self.f1 = 0.0
        self.solution = None
        self.info = None
        self.dneta_dp = None
        self.df1_dp = None
        self.dk1_dp = None
        self.records = []
        
        # Initialise the nonlinear optimizer.
        inf = 10e19
        dimension = self.sym.dimension
        n_constraints = 2
        lb = 1e-4 * np.ones(dimension) 
        ub = np.ones(dimension)
        cl = np.array((-inf, f0))
        cu = np.array((k0, inf))
        self.nlp = ipopt.problem(n=dimension, m=n_constraints, 
                                 problem_obj=self, lb=lb, ub=ub, cl=cl, cu=cu)
    
        # Configure the nonlinear optimizer.
        log_file = ''.join((self.dir, '/', self.tag, '-log.txt')).encode()
        #self.nlp.addOption(b'derivative_test', b'second-order')
        self.nlp.addOption(b'mu_strategy', b'adaptive')
        self.nlp.addOption(b'max_iter', max_iter)
        self.nlp.addOption(b'tol', 1e-8)
        self.nlp.setProblemScaling(obj_scaling=1e6)
        self.nlp.addOption(b'nlp_scaling_method', b'user-scaling')
        self.nlp.addOption(b'output_file', log_file)
        
        
    ###########################################################################
    # Ipopt callbacks.
    ###########################################################################
    def objective(self, xs):
        
        logger.info('calculate objective')
        self.analysis(xs)
        return self.neta1
    
    
    def gradient(self, xs):
        
        logger.info('calculate gradient')
        self.analysis(xs)
        return self.dneta1_dp
    
    
    def constraints(self, xs):
        
        logger.info('calculate constraints')
        self.analysis(xs)
        return np.array((self.k1, self.f1))
    
    
    def jacobian(self, xs):

        logger.info('calculating jacobian')
        self.analysis(xs)
        return np.concatenate((self.dk1_dp, self.df1_dp))
    
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        
        self.records.append((iter_count, obj_value, self.k1, self.f1))
        v = (iter_count, obj_value)
        print('>>> finished iteration %d: objective = %g' % v)
        if self.debug == True:  
            t0 = time.time()
            #self.analyser.plot_mode(0)
            self.analyser.plot_densities()
            t1 = time.time()
            logger.info('Time to print images (s): %g' % (t1-t0))
    
    
    ###########################################################################
    # Console logging functions.
    ###########################################################################
    def to_console_init(self):
        
        print('--- Dynamic AFM Cantilever Optimization ---')
        print('Total number of variables: %d' % self.fem.n_elem)
        
        
    def to_console_final(self):
        
        g = self.info['g']
        ov = self.info['obj_val']
        msg = self.info['status_msg'].decode('utf-8') 

        print(''.join(('\n', msg, '\n')))
        print('Timing for execution                (s)  : %g' % self.execution_time)
        print('Stiffness at the optimal solution   (N/m): %s' % g[0])
        print('Frequency at the optimal solution   (Hz) : %s' % g[1])
        print('Sensitivity at the optimal solution (C/m): %s' % ov)
        
        print('\n--- Solution Analysis ---')
        fn = ''.join((self.dir, '/', self.tag, '-image.png'))
        self.analyser.plot_densities(fn)
        self.analyser.identify_modal_parameters()
        
        
    ###########################################################################
    # Execution and analysis.
    ###########################################################################
    def execute(self):

        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        if os.path.isfile(fn):
            print('Solution already exists. Delete .npy file to continue.')
            return
        
        self.to_console_init()
        t0 = time.time()
        self.solution, self.info = self.nlp.solve(self.x0)
        self.save_solution()
        self.save_records()
        t1 = time.time()
        self.execution_time = t1 - t0
        self.to_console_final()
        
    
    def analysis(self, xs):
        
        if np.array_equal(xs, self.xs_prev) == True:
            return

        self.xs_prev = xs.copy()
        t0 = time.time()
        
        # Structure regularization.
        x1 = self.sym.execute(xs)
        if self.to_filter == True:
            dens1 = self.density_filter.execute(x1) 
        else:
            dens1 = xs
            
        if self.to_project == True:
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
        
        logger.info('Cantilever neg. charge sens is (C/m): %g' % self.neta1)
        logger.info('Cantilever stiffness is (N/m):        %g' % self.k1)
        logger.info('Cantilever frequency is (Hz):         %g' % self.f1)
        
        # Sensitivity Analysis.
        self.dneta1_dp = self.fem.charge_grad(lam1, phi1, wtip1, charge1, guu)
        self.dk1_dp = self.fem.stiff_grad(lam1, phi1, wtip1, self.k1, guu)
        self.df1_dp = self.fem.freq_grad(lam1, phi1)  / (4.0 * np.pi * np.sqrt(lam1))
        
        if self.to_project == True:
            dprojection = self.projection.sensitivity(dens1)
            self.dneta1_dp = self.dneta1_dp @ dprojection
            self.dk1_dp = self.dk1_dp @ dprojection
            self.df1_dp = self.df1_dp @ dprojection
            
        if self.to_filter == True:
            dfilter = self.density_filter.sensitivity()
            self.dneta1_dp = self.dneta1_dp @ dfilter
            self.dk1_dp = self.dk1_dp @ dfilter
            self.df1_dp = self.df1_dp @ dfilter
            
        dsym = self.sym.sensitivity()
        self.dneta1_dp = self.dneta1_dp @ dsym
        self.dk1_dp = self.dk1_dp @ dsym
        self.df1_dp = self.df1_dp @ dsym
            
        t1 = time.time()
        logger.info('Timing for analysis (s): %g' % (t1-t0))
        
        
    ###########################################################################
    # Load/save data.
    ###########################################################################
    def save_solution(self):
        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        np.save(fn, opt.solution)    
        fn = ''.join((self.dir, '/', self.tag, '-design.txt'))
        np.savetxt(fn, self.solution)
            
    
    def load_solution(self):
        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        return np.load(fn) 
    
    
    def save_records(self):
        header = 'Iteration, Objective Value, Stiffness, Frequency'
        fn = ''.join((self.dir, '/', self.tag, '-records.txt'))
        data = np.array(self.records)
        np.savetxt(fn, data, delimiter=',', header=header)
        
        
    def select_cantilever(self):
        if self.cantilever_key == 'InitialCantileverFixedTip':
            return cantilevers.InitialCantileverFixedTip()
        else:
            return cantilevers.InitialCantileverHigherFreq()

    
if __name__ == '__main__':
    opt = main()