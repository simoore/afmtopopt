import pprint
from afmtopopt import TopologyOptimizer, load_parameters

def main():
    
    batch = ['solutions/problem1.yaml',
             'solutions/problem2.yaml']
    opt_list = []
    
    for b in batch:
        params = load_parameters(b)
        print('-- Parameters --\n')
        pprint.pprint(params)
        print()
    
        optimizer = TopologyOptimizer(params)
        optimizer.cantilever.to_console()
    
        print('\n--- Initial Analysis ---')
        optimizer.analyser.plot_densities()
        optimizer.analyser.identify_modal_parameters()
    
        optimizer.execute()
        opt_list.append(optimizer)
        
    return opt_list

if __name__ == '__main__':
    opt = main()