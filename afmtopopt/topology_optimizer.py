import time
import numpy as np
import ipopt
import logging
import os


from . import materials
from . import cantilevers
from . import projection
from . import density_filter
from . import analysers
from . import symmetry
from .laminate_analysis import LaminateAnalysis


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopologyOptimizer(object):
    
    def __init__(self, params):
                
        beta = params['beta']
        k0 = params['k0']
        f0 = params['f0']
        rmin = params['rmin']
        max_iter = params['max_iter']
        self.tag = params['tag']
        self.debug = params['debug']
        self.cantilever_key = params['cantilever']
        self.dir = params['dir']
        to_connect = params['to_connect']
        obj_scale = params['obj_scale']
        pmu = params['pmu']
        
        self.material = materials.PiezoMumpsMaterial()
        self.cantilever = self.select_cantilever()
        self.la = LaminateAnalysis(self.cantilever, self.material, to_connect, pmu)
        self.analyser = analysers.CantileverAnalyser(self.la.fem)
        
        self.sym = symmetry.Symmetry(self.la.fem)
        self.density_filter = density_filter.DensityFilter(self.la.fem, rmin)
        self.projection = projection.Projection(beta)
        self.x0 = self.sym.initial(self.la.fem.mesh.get_densities())
        
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
        self.nlp = ipopt.problem(n=self.sym.dimension, 
                                 m=2, 
                                 problem_obj=self, 
                                 lb=1e-4 * np.ones(self.sym.dimension), 
                                 ub=np.ones(self.sym.dimension), 
                                 cl=np.array((-inf, f0)), 
                                 cu=np.array((k0, inf)))
    
        # Configure the nonlinear optimizer.
        log_file = ''.join((self.dir, '/', self.tag, '-log.txt')).encode()
        self.nlp.addOption(b'max_iter', max_iter)
        self.nlp.addOption(b'tol', 1e-5)
        self.nlp.addOption(b'acceptable_tol', 1e-3)
        self.nlp.addOption(b'obj_scaling_factor', obj_scale)
        self.nlp.addOption(b'output_file', log_file)
        self.nlp.addOption(b'expect_infeasible_problem', b'yes')
        
        
    ###########################################################################
    # Ipopt callbacks.
    ###########################################################################
    def objective(self, xs):
        
        logger.info('calculate objective')
        self.analysis(xs)
        return self.la.neta1
    
    
    def gradient(self, xs):
        
        logger.info('calculate gradient')
        self.analysis(xs)
        return self.dneta1_dp
    
    
    def constraints(self, xs):
        
        logger.info('calculate constraints')
        self.analysis(xs)
        return np.array((self.la.k1, self.la.f1))
    
    
    def jacobian(self, xs):
        
        logger.info('calculating jacobian')
        self.analysis(xs)
        return np.concatenate((self.dk1_dp, self.df1_dp))
    
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        
        self.records.append((iter_count, obj_value, self.la.k1, self.la.f1))
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
        print('Total number of variables: %d' % self.la.fem.mesh.n_elem)
        
        
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
        
        
    def log_iteration(self):
        
        m1 = 'Cantilever neg. charge sens is (C/m): %g'
        m2 = 'Cantilever stiffness is (N/m):        %g'
        m3 = 'Cantilever frequency is (Hz):         %g'
        
        logger.info(m1 % self.la.neta1)
        logger.info(m2 % self.la.k1)
        logger.info(m3 % self.la.f1)
        
        
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
        
        # Check if previous solution is the current one to avoid duplicating
        # the analysis.
        if np.array_equal(xs, self.xs_prev) == True:
            return

        self.xs_prev = xs.copy()
        t0 = time.time()
        
        # Structure regularization.
        x1 = self.sym.execute(xs)
        x2 = self.density_filter.execute(x1)
        x3 = self.projection.execute(x2)
        
        # Finite Element Analysis.
        self.la.execute_analysis(x3)
        self.log_iteration()
        
        # Sensitivity of structural regularization.
        dsym = self.sym.sensitivity()
        dfilt = self.density_filter.sensitivity()
        dproj = self.projection.sensitivity(x2)
        self.dneta1_dp = self.la.dneta1 @ dproj @ dfilt @ dsym
        self.dk1_dp = self.la.dk1 @ dproj @ dfilt @ dsym
        self.df1_dp = self.la.df1 @ dproj @ dfilt @ dsym

        t1 = time.time()
        logger.info('Timing for analysis (s): %g' % (t1-t0))
        
        
    ###########################################################################
    # Load/save data.
    ###########################################################################
    def save_solution(self):
        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        np.save(fn, self.solution)    
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
        
        funcs =  {'InitialCantileverFixedTip': cantilevers.InitialCantileverFixedTip,
                  'InitialCantileverRectangular': cantilevers.InitialCantileverRectangular,
                  'InitialCantileverRectangularStep': cantilevers.InitialCantileverRectangularStep, 
                  'InitialCantileverHigherFreq': cantilevers.InitialCantileverHigherFreq,
                  'StandardA': cantilevers.StandardA,
                  'StandardB': cantilevers.StandardB,
                  'StandardC': cantilevers.StandardC,
                  'StandardD': cantilevers.StandardD,
                  'StandardE': cantilevers.StandardE,
                  'StandardF': cantilevers.StandardF,
                  'StandardG': cantilevers.StandardG,
                  'StandardH': cantilevers.StandardH,
                  'StandardI': cantilevers.StandardI,
                  'StandardJ': cantilevers.StandardJ}

        if self.cantilever_key not in funcs:
            raise ValueError('Non-existent cantilever class.')
        return funcs[self.cantilever_key]()