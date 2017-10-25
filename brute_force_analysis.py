import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    bfa = BruteForceAnalysis()
    bfa.charge_vs_stiffness()
    #bfa.charge_vs_length_ratio()
    #bfa.charge_vs_width_fixed_length()
    #bfa.charge_vs_area_fixed_length()
    #bfa.charge_vs_area()
    bfa.charge_vs_freq()
    #bfa.charge_vs_freq_and_stiffness()
    #bfa.charge_vs_ratios()
    #bfa.freq_vs_ratios()
    #bfa.stiffness_vs_ratios()
    bfa.best_cantilever()
    bfa.equality_hypothesis()
    
class BruteForceAnalysis(object):
    
    def __init__(self):
        self.data = None
        self.filename = 'brute-force-b.txt'
        self.load_data()
        self.length = np.array([d[0] for d in self.data])
        self.width = np.array([d[1] for d in self.data])
        self.tip_length = np.array([d[2] for d in self.data])
        self.tip_width = np.array([d[3] for d in self.data])
        self.freq1 = np.array([d[4] for d in self.data])
        self.freq2 = np.array([d[5] for d in self.data])
        self.eta1 = -1e6 * np.array([d[6] for d in self.data])
        self.eta2 = -1e6 * np.array([d[7] for d in self.data])
        self.k1 = np.array([d[8] for d in self.data])
        self.k2 = np.array([d[9] for d in self.data])
        self.length_ratio = self.tip_length / (self.length - self.tip_length)
        self.width_ratio = self.tip_width / self.width
        
        
    def load_data(self):
        with open(self.filename, 'rb') as fp:
            self.data = pickle.load(fp)
        
        
    def charge_vs_stiffness(self):
        fig, ax = plt.subplots()
        for i in range(6, 14):
            index = self.length == i
            k1i = self.k1[index]
            eta1i = self.eta1[index]
            ax.scatter(k1i, eta1i, label=''.join(('length ', str(i))))
        ax.set_xlabel('Stiffness N/m')
        ax.set_ylabel('Charge Sensitivity C/um')
        ax.legend()
        plt.show()
        
        
    def charge_vs_length_ratio(self):
        fig, ax = plt.subplots()
        for i in range(6, 14):
            index = self.length == i
            lr = self.length_ratio[index]
            eta1i = self.eta1[index]
            ax.scatter(lr, eta1i, label=''.join(('length ', str(i))))
        ax.set_xlabel('Length Ratio')
        ax.set_ylabel('Charge Sensitivity C/um')
        ax.legend()
        plt.show()
        
        
    def charge_vs_width_fixed_length(self):
        fixed_length = 12
        index = self.length == fixed_length
        eta = self.eta1[index]
        width = self.width[index]
        wr = self.width_ratio[index]
        fig, ax = plt.subplots()
        ax.scatter(width, eta)
        
        index = wr == 1
        e2 = eta[index]
        w2 = width[index]
        ax.scatter(w2, e2)
        
        ax.set_title('Cantilevers of length 120um')
        ax.set_xlabel('Width 10x um')
        ax.set_ylabel('Charge Sensitivity C/um')
        plt.show()
        
        
    def charge_vs_area_fixed_length(self):
        fixed_length = 12
        index = self.length == fixed_length
        eta = self.k1[index]
        width = self.width[index]
        tip_width = self.tip_width[index]
        tip_length = self.tip_length[index]
        area = tip_width * tip_length + (fixed_length - tip_length) * width
        fig, ax = plt.subplots()
        ax.scatter(area, eta)
        ax.set_title('Cantilevers of length 120um')
        ax.set_xlabel('Area 100um$^2$')
        ax.set_ylabel('Charge Sensitivity C/um')
        plt.show()
        
        
    def charge_vs_area(self):
        area = self.tip_width * self.tip_length + (self.length - self.tip_length) * self.width
        fig, ax = plt.subplots()
        for i in range(6, 14):
            index = self.length == i
            ar = area[index]
            eta1i = self.eta1[index]
            ax.scatter(ar, eta1i, label=''.join(('length ', str(i))))
        ax.set_xlabel('Area 100x um$^2$')
        ax.set_ylabel('Charge Sensitivity C/um')
        ax.legend()
        plt.show()
        
        
    def charge_vs_freq(self):
        fig, ax = plt.subplots()
        for i in range(6, 14):
            index = self.length == i
            freq1i = self.freq1[index]
            eta1i = self.eta1[index]
            ax.scatter(freq1i, eta1i, label=''.join(('length ', str(i))))
        ax.set_xlabel('Frequency Hz')
        ax.set_ylabel('Charge Sensitivity C/um')
        ax.legend()
        plt.show() 
        
        
    def charge_vs_freq_and_stiffness(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.k1, self.freq1, self.eta1)
        plt.show()
        
    def charge_vs_ratios(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.length_ratio, self.width_ratio, self.eta1)
        ax.set_xlabel('Length Ratio')
        ax.set_ylabel('Width Ratio')
        ax.set_zlabel('Charge Sensitivity')
        plt.show()
        
    def freq_vs_ratios(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.length_ratio, self.width_ratio, self.freq1)
        ax.set_xlabel('Length Ratio')
        ax.set_ylabel('Width Ratio')
        ax.set_zlabel('Frequency')
        plt.show()
        
        
    def stiffness_vs_ratios(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.length_ratio, self.width_ratio, self.k1)
        ax.set_xlabel('Length Ratio')
        ax.set_ylabel('Width Ratio')
        ax.set_zlabel('Stiffness')
        plt.show()
        
        
    def best_cantilever(self):
        best_index = 0
        best_eta1 = 0
        for i in range(len(self.data)):
            if self.k1[i] < 5000 and self.freq1[i] > 1500000:
                if self.eta1[i] > best_eta1:
                    best_index = i
                    best_eta1 = self.eta1[i]
        print(best_index)
        print('Best eta1 is: %g' % best_eta1)
        print('Best k1 is: %g' % self.k1[best_index])
        print('Best freq1 is: %g' % self.freq1[best_index])
        
        
    def equality_hypothesis(self):
        for be, bk, bw in zip(self.eta1, self.k1, self.freq1):
            print('---------')
            print('%g %g %g' % (be, bk, bw))
            constraints = np.logical_and(self.k1 < bk, self.freq1 > bw)
            index = np.logical_and(self.eta1 > be, constraints)
            for e, k, w in zip(self.eta1[index], self.k1[index], self.freq1[index]):
                print('%g %g %g' % (e, k, w))
        print('Done')
    
    
        
        
if __name__ == '__main__':
    main()
            