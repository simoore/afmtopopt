import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special

class Gaussian(object):
    
    def __init__(self, sigma, xc, yc, xa, xb, ya, yb):
        self._sigma = sigma
        self._xc = xc
        self._yc = yc
        self._xa = xa
        self._xb = xb
        self._ya = ya
        self._yb = yb
        
    
    def evaluate(self, x, y):
        xx = x - self._xc
        yy = y - self._yc
        normalization = 1 / (2 * np.pi * self._sigma ** 2) 
        exponent = -(xx ** 2 + yy ** 2) / (2 * self._sigma ** 2)
        gau = normalization * np.exp(exponent)
        return gau
    
    
    def numerical_integration(self, dx, dy):
        da = dx * dy
        x = np.arange(self._xa, self._xb, dx)
        y = np.arange(self._ya, self._yb, dy)
        x, y = np.meshgrid(x, y)
        x = np.ravel(x)
        y = np.ravel(y)
        eval_ = da * sum((self.evaluate(x,y) for x, y in zip(x, y)))
        return eval_
    
    
    def weighted_numerical_integration(self, func, dx, dy):
        da = dx * dy
        xx = np.arange(self._xa, self._xb, dx)
        yy = np.arange(self._ya, self._yb, dy)
        xx, yy = np.meshgrid(xx, yy)
        xx = np.ravel(xx)
        yy = np.ravel(yy)
        eval_ = 0
        for x, y in zip(xx, yy):
            eval_ += self.evaluate(x, y) * func(x, y)
        return da * eval_
    
    
class BasicMesh(object):
    
    def __init__(self, nelx, nely, a, b):
        self.nelx = nelx
        self.nely = nely
        self.elems = [[None for _ in range(nely)] for _ in range(nelx)]
        index = np.ndindex(nelx, nely)
        for i, j in index:
            self.elems[i][j] = BasicElement(i, j, a, b)
            
            
    def set_coefficients(self, gau):
        index = np.ndindex(self.nelx, self.nely)
        for i, j in index:
            #self.elems[i][j].set_coefficients(gau)
            self.elems[i][j].generate_gaussian_element(gau._xc, gau._yc, gau._sigma)
            
        
        
        
class BasicElement(object):
    
    def __init__(self, i, j, a, b):
        self.coefficients = [0 for i in range(4)]
        self.eval_coeffs = [0 for i in range(4)]
        self.nodes = (4, 4, 4, 4)
        self.wave = 0
        self.wave_eval = 0
        self.x0 = 2 * i * a
        self.y0 = 2 * j * b
        self.a = a
        self.b = b
        self.jacobian = a * b
        
    def set_coefficients(self, gau):
        
        for i in range(4):
            def f(xn, yn):
                n = BasicElement.shapes((xn, yn), i)
                f = n * gau.evaluate(self.x0 + self.a*xn, self.y0 + self.b*yn)
                return f
            c = integrate.dblquad(f, -1, 1, lambda x: -1, lambda x: 1)
            self.coefficients[i] = self.jacobian * c[0]
        self.wave = sum(self.coefficients)
        
        
    def generate_gaussian_element(self, xmid, ytip, sigma):
        k1 = 1 / (4 * np.pi)
        #xdif = xmid - 2 * a * self.x0
        #ydif = ytip - 2 * b * self.y0
        xdif = xmid - self.x0
        ydif = ytip - self.y0
        xa = (xdif - self.a) / (np.sqrt(2) * sigma)
        xb = (xdif + self.a) / (np.sqrt(2) * sigma)
        ya = (ydif - self.b) / (np.sqrt(2) * sigma)
        yb = (ydif + self.b) / (np.sqrt(2) * sigma)
        
        int1 = np.sqrt(np.pi/4) * (special.erf(xb) - special.erf(xa))
        int2 = sigma / np.sqrt(2) * (np.exp(-xb * xb) - np.exp(-xa * xa))
        int3 = np.sqrt(np.pi/4) * (special.erf(yb) - special.erf(ya))
        int4 = sigma / np.sqrt(2) * (np.exp(-yb * yb) - np.exp(-ya * ya))
        
        xi_a = [x/self.a for x in (-1, -1, 1, 1)]
        yi_b = [y/self.b for y in (-1, 1, 1, -1)]
        ge = [0 for i in range(4)]
        for i in range(4):
            k2 = (1 - xi_a[i] * xdif) * int1 - xi_a[i] * int2
            k3 = (1 - yi_b[i] * ydif) * int3 - yi_b[i] * int4
            ge[i] = k1 * k2 * k3
        #print(ge)
        self.eval_coeffs = ge
        self.wave_eval = sum(ge)
            
        
    def shapes(point, index):
        '''The index refers to a node in the normalized element.
        index = 0 : node sw
        index = 1 : node se
        index = 2 : node ne
        index = 3 : node nw
        '''
        xi, eta = point
        xi_sign = [-1, 1, 1, -1]
        eta_sign = [-1, -1, 1, 1]
        xs, es = xi_sign[index], eta_sign[index]
        n = 0.25 * (1 + xs * xi) * (1 + es * eta)
        return n    
 
# TEST ONE ELEMENT
gau = Gaussian(0.3, xc=20, yc=40, xa=0, xb=40, ya=0, yb=80)       
element = BasicElement(20, 40, 0.5, 0.5)
element.set_coefficients(gau)
element.generate_gaussian_element(gau._xc, gau._yc, gau._sigma)
print(element.coefficients)
print(element.eval_coeffs)
print(sum(element.coefficients))
print(sum(element.eval_coeffs))

#gau = Gaussian(0.3, xc=20, yc=40, xa=0, xb=40, ya=0, yb=80)
#mesh = BasicMesh(40, 80, 0.5, 0.5)
#mesh.set_coefficients(gau)
    

#print(gau.numerical_integration(0.1, 0.1))

#func = lambda x, y: 4
#print(gau.weighted_numerical_integration(func, 0.1, 0.1))


#ee = integrate.dblquad(gau.evaluate, 0, 80, lambda x: 0, lambda x: 40)
#print(ee)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#x = np.arange(0, 40, 0.1)
#y = np.arange(0, 80, 0.1)
#X, Y = np.meshgrid(x, y)
#zs = np.array([gau.evaluate(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
#Z = zs.reshape(X.shape)
#ax.plot_surface(X, Y, Z)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#x = np.arange(0, 40, 1)
#y = np.arange(0, 80, 1)
#X, Y = np.meshgrid(x, y)
#index = np.ndindex(40, 80)
#zs = np.array([sum(mesh.elems[x][y].coefficients[0]) for x, y in index])
#Z = zs.reshape(X.shape)
#ax.plot_surface(X, Y, Z)
#        
        
        
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#x = np.arange(0, 40, 1)
#y = np.arange(0, 80, 1)
#X, Y = np.meshgrid(x, y)
#index = zip(np.ravel(X), np.ravel(Y))
##zs = np.array([gau.evaluate(mesh.elems[x][y].x0, mesh.elems[x][y].y0) for x, y in index])
#zs = np.array([mesh.elems[x][y].wave for x, y in index])
#Z = zs.reshape(X.shape)
#ax.plot_surface(X, Y, Z)
#print(sum(zs))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#x = np.arange(0, 40, 1)
#y = np.arange(0, 80, 1)
#X, Y = np.meshgrid(x, y)
#index = zip(np.ravel(X), np.ravel(Y))
#zs = np.array([mesh.elems[x][y].wave_eval for x, y in index])
#Z = zs.reshape(X.shape)
#ax.plot_surface(X, Y, Z)
#print(sum(zs))
        
    
    


