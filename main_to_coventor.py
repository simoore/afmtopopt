import numpy as np
import density_filter
import symmetry
import projection
import laminate_fem
import materials
import cantilevers
from ruamel.yaml import YAML
import os
import analysers


def main():

    # Load configuration for solution.
    config_file = 'solutions/std-e12.yaml'
    print(config_file)
    cantilever, material, to_connect, pmu = generate_cantilever(config_file)
    
    # Analysis of new solution.
    fem = laminate_fem.LaminateFEM(cantilever, material, to_connect, pmu)
    analyser = analysers.CantileverAnalyser(fem)
    to_coventor(cantilever)
    
    # Visual check of solution.
    analyser.plot_densities()
    #analyser.identify_modal_parameters()
    #analyser.plot_mode(4)
    
    
#def main():
#    topology = np.load('solutions/best_topology.npy')
#    nelx, nely = topology.shape
#    cantilever = cantilevers.RectangularCantilever(5e-6, 5e-6, nelx, nely)
#    cantilever.topology = topology
#    cantilever.densities = topology
#    print(cantilever.xtip)
#    print(cantilever.ytip)
#    
#    material = materials.PiezoMumpsMaterial()
#    fem = laminate_fem.LaminateFEM(cantilever, material)
#    analyser = analysers.CantileverAnalyser(fem)
#    to_coventor(cantilever)
#    
#    analyser.plot_densities()
#    analyser.identify_modal_parameters()
#    analyser.plot_mode(0)
#    analyser.plot_mode(1)
#    analyser.plot_mode(2)
    
    
def generate_cantilever(config_file):
    
    # Load parameters for solution.
    params = load_parameters(config_file)
    
    beta = params['beta']
    rmin = params['rmin']
    tag = params['tag']
    cantilever_key = params['cantilever']
    dir_ = params['dir']
    to_connect = params['to_connect']
    pmu = params['pmu']
    
    # Initialize structural regularization operators.
    material = materials.PiezoMumpsMaterial()
    cantilever = select_cantilever(cantilever_key)
    fem = laminate_fem.LaminateFEM(cantilever, material, to_connect, pmu)
    
    # original=True for std-axx, std-bxx, std-cxx, std-dxx, else False.
    sym = symmetry.Symmetry(fem, original=False)
    density_filter_ = density_filter.DensityFilter(fem, rmin)
    projection_ = projection.Projection(beta)
    analyser = analysers.CantileverAnalyser(fem)

    # Load the solution and produce the structure.
    filename = ''.join((dir_, '/', tag, '-design.npy'))
    solution = np.load(filename)
    x1 = sym.execute(solution)
    x2 = density_filter_.execute(x1)
    x3 = projection_.execute(x2)
    fem.assemble(x3)
    
    img_name = ''.join(('plots/', tag, '.png'))
    analyser.plot_densities(img_name)
    analyser.identify_modal_parameters()
    
    # Generate new topology.
    nelx, nely = cantilever.topology.shape
    data = np.zeros((nelx, nely))
    for e, p in zip(fem.mesh.elements, fem.density_penalty):
        if p > 0.5:
            data[e.i, e.j] = 1
    cantilever.topology = data
    cantilever.densities = data
    
    return cantilever, material, to_connect, pmu


def to_coventor(cantilever):
    
    dx = 2e6 * cantilever.a
    dy = 2e6 * cantilever.b
    nelx, nely = cantilever.topology.shape

    with open('coventor.txt', 'w') as the_file:
        the_file.write('cat:layer Soi\n')
        for i in range(nelx):
            for j in range(nely):
                if cantilever.topology[i, j] == 1:
                    x1 = i * dx
                    y1 = j * dy
                    x2 = x1 + dx
                    y2 = y1 + dy
                    tup = (x1, y1, x2, y2)
                    the_file.write('cat:rectangle %g %g %g %g\n' % tup)
 

def load_parameters(filename):
    
    with open(filename, 'r') as f:
        yaml = YAML(typ='safe')
        params = yaml.load(f)
        params['tag'] = os.path.splitext(os.path.basename(f.name))[0]
        params['dir'] = os.path.dirname(f.name)
        return params
           
            
def select_cantilever(cantilever_key):
    if cantilever_key == 'InitialCantileverFixedTip':
        return cantilevers.InitialCantileverFixedTip()
    elif cantilever_key == 'InitialCantileverRectangular':
        return cantilevers.InitialCantileverRectangular()
    elif cantilever_key == 'InitialCantileverRectangularStep':
        return cantilevers.InitialCantileverRectangularStep()
    elif cantilever_key == 'StandardA':
        return cantilevers.StandardA()
    elif cantilever_key == 'StandardB':
        return cantilevers.StandardB()
    elif cantilever_key == 'StandardC':
        return cantilevers.StandardC()
    elif cantilever_key == 'StandardD':
        return cantilevers.StandardD()
    else:
        return cantilevers.InitialCantileverHigherFreq()
            
            
if __name__ == '__main__':
    main()
