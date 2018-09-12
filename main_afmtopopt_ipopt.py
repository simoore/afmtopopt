import pprint
from afmtopopt import TopologyOptimizer, load_parameters

def main():
    
    batch = ['_solutions/lgr-h01.yaml',
             '_solutions/lgr-h02.yaml',
             '_solutions/lgr-h03.yaml',
             '_solutions/lgr-h04.yaml',
             '_solutions/lgr-h05.yaml',
             '_solutions/lgr-h06.yaml',
             '_solutions/lgr-h07.yaml',
             '_solutions/lgr-h08.yaml',
             '_solutions/lgr-h09.yaml',
             '_solutions/lgr-h10.yaml',
             '_solutions/lgr-h11.yaml',
             '_solutions/lgr-h12.yaml']
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