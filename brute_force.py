import pickle
import time
import numpy as np
import os.path
import cantilevers
import materials
import finite_element
import gaussian


def main():
    search = BruteForce()
    search.execute()
    return search


class BruteForce(object):
    
    def __init__(self):
        self.max_width = 14
        self.max_length = 14
        self.data = []
        self.filename = 'brute-force-b.txt'
        
    
    def dimension_generator(self):
        for length in range(4, self.max_length + 1):
            for width in range(4, self.max_width + 1):
                for tip_length in range(0, length):
                    start = 4 if (width % 2 == 0) else 3
                    for tip_width in range(start, width + 1, 2):
                        yield (length, width, tip_length, tip_width)
        
        
    def execute(self):
        
        if os.path.isfile(self.filename) == True:
            print('Data already exists, delete before execution')
            self.load_data()
            return
        
        t0 = time.time()
        self.data = []
        material = materials.PiezoMumpsMaterial()
        dimensions = self.dimension_generator()
        
        for len_, wid, tl, tw in dimensions:
            
            cantilever = cantilevers.SteppedCantilever(wid, len_, tw, tl)
            xtip, ytip, sigma = cantilever.xtip, cantilever.ytip_ub, 0.1
            fem = finite_element.LaminateFEM(cantilever, material)
            gau = gaussian.Gaussian(fem.get_mesh(), xtip, ytip, sigma)
            w, v = fem.modal_analysis(2)
            
            kuu = fem.get_stiffness_matrix()
            kuv = fem.get_piezoelectric_matrix()
            guu = gau.get_operator()
        
            phi1 = v[:, [0]]
            phi2 = v[:, [1]]
            wtip1 = np.asscalar(guu @ phi1)
            wtip2 = np.asscalar(guu @ phi2)
            charge1 = np.asscalar(kuv.T @ phi1)
            charge2 = np.asscalar(kuv.T @ phi2)
            freq1 = np.sqrt(w[0]) / 2.0 / np.pi
            freq2 = np.sqrt(w[1]) / 2.0 / np.pi
            eta1 = charge1 / wtip1
            eta2 = charge2 / wtip2
            k1 = np.asscalar(phi1.T @ kuu @ phi1 / wtip1 ** 2)
            k2 = np.asscalar(phi2.T @ kuu @ phi2 / wtip2 ** 2)
            record = (len_, wid, tl, tw, freq1, freq2, eta1, eta2, k1, k2)
            self.data.append(record)
            
        t1 = time.time()
        print('Analysis took %g (s)' % (t1 - t0))
        self.save_data()
        
        
    def save_data(self):
        with open(self.filename, 'wb') as fp:
            pickle.dump(self.data, fp)
            
            
    def load_data(self):
        with open(self.filename, 'rb') as fp:
            self.data = pickle.load(fp)


if __name__ == '__main__':
    search = main()