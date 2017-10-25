"""This is the tutorial example of a nonlinear constrained optimization solved
using NLopt. The website this example is from is:
    
    http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial
    
It is a two dimensional optimization problem given by:
    
    min sqrt(x_2)
    s.t. x_2 >= 0
         x_2 >= (a_1 * x_1 + b_1)^3
         x_2 >= (a_2 * x_1 + b_2)^3
    
For this problem let a_1=2, b_1=0, a_2=-1, and b_2=1. The optimum solution 
occurs when the cost function is 0.544331 with solution x_1=0.3333, and 
x_2=0.296296.
"""
        
import nlopt
import numpy as np


def myfunc(x, grad):
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / np.sqrt(x[1])
    return np.sqrt(x[1])


def myconstraint(x, grad, a, b):
    if grad.size > 0:
        grad[0] = 3 * a * (a*x[0] + b)**2
        grad[1] = -1.0
    return (a*x[0] + b)**3 - x[1]


opt = nlopt.opt(nlopt.LD_MMA, 2)
opt.set_lower_bounds([-float('inf'), 0])
opt.set_min_objective(myfunc)
opt.add_inequality_constraint(lambda x, grad: myconstraint(x,grad, 2, 0), 1e-8)
opt.add_inequality_constraint(lambda x, grad: myconstraint(x,grad, -1, 1), 1e-8)
opt.set_xtol_rel(1e-4)
x = opt.optimize([1.234, 5.678])
minf = opt.last_optimum_value()

print('optimum at ', x[0], x[1])
print('minimum value = ', minf)
print('result code = ', opt.last_optimize_result())