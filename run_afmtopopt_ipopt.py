import pprint
from afmtopopt import TopologyOptimizer, load_parameters

def main():
    
    batch = ['_solutions/lng-a01.yaml',
             '_solutions/lng-a02.yaml',
             '_solutions/lng-a03.yaml',
             '_solutions/lng-a04.yaml',
             '_solutions/lng-a05.yaml',
             '_solutions/lng-a06.yaml',
             '_solutions/lng-a07.yaml',
             '_solutions/lng-a08.yaml',
             '_solutions/lng-a09.yaml',
             '_solutions/lng-a10.yaml',
             '_solutions/lng-a11.yaml',
             '_solutions/lng-a12.yaml',
             '_solutions/lng-b01.yaml',
             '_solutions/lng-b02.yaml',
             '_solutions/lng-b03.yaml',
             '_solutions/lng-b04.yaml',
             '_solutions/lng-b05.yaml',
             '_solutions/lng-b06.yaml',
             '_solutions/lng-b07.yaml',
             '_solutions/lng-b08.yaml',
             '_solutions/lng-b09.yaml',
             '_solutions/lng-b10.yaml',
             '_solutions/lng-b11.yaml',
             '_solutions/lng-b12.yaml']
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