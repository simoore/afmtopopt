import pprint
from afmtopopt import TopologyOptimizer, load_parameters

def main():
    
    batch = ['_solutions/wid-a01.yaml',
             '_solutions/wid-a02.yaml',
             '_solutions/wid-a03.yaml',
             '_solutions/wid-a04.yaml',
             '_solutions/wid-a05.yaml',
             '_solutions/wid-a06.yaml',
             '_solutions/wid-a07.yaml',
             '_solutions/wid-a08.yaml',
             '_solutions/wid-a09.yaml',
             '_solutions/wid-a10.yaml',
             '_solutions/wid-a11.yaml',
             '_solutions/wid-a12.yaml',
             '_solutions/wid-b01.yaml',
             '_solutions/wid-b02.yaml',
             '_solutions/wid-b03.yaml',
             '_solutions/wid-b04.yaml',
             '_solutions/wid-b05.yaml',
             '_solutions/wid-b06.yaml',
             '_solutions/wid-b07.yaml',
             '_solutions/wid-b08.yaml',
             '_solutions/wid-b09.yaml',
             '_solutions/wid-b10.yaml',
             '_solutions/wid-b11.yaml',
             '_solutions/wid-b12.yaml']
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