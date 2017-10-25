import numpy as np


class Cantilever(object):
    """Cantilevers are defined by three parameters. (topology) a binary matrix
    indicating the existence of elements in a rectangular mesh, (a) half the 
    width of the element, (b) have the length of the element. The first index
    is the x-coordinate and the second index is the y-coordinate. Elements
    along the x-axis (y == 0) are on the boundary.
    """
    
    def __init__(self, topology, a, b, xtip, ytip, densities):
        """xtip and ytip are in um.
        """
        self.topology = topology
        self.a = a
        self.b = b
        self.xtip = xtip
        self.ytip = ytip
        self.densities = densities

        
class BJNCantilever(Cantilever):

    def __init__(self):
        
        a = 5e-6
        b = 5e-6
        xtip = 250
        ytip = 795
        self.nr = round(500e-6 / (2 * a))
        self.nc = round(800e-6 / (2 * b))
        topology = np.zeros((self.nr, self.nc))
        densities = topology
        for r, c in np.ndindex(self.nr, self.nc):
            if self.in_set(r, c):
                topology[r, c] = 1
        super().__init__(topology, a, b, xtip, ytip, densities)
                    
            
    def in_set(self, r, c):
        
        if c < 0.5 * self.nc:
            return True
        elif r >= 0.4 * self.nr and r < 0.6 * self.nr:
            return True
        else:
            return False    
        
        
class SteppedCantilever(Cantilever):
    
    def __init__(self, width, length, tip_width, tip_length):
        """(width - tip_width) must be even.
        """
        
        a = 5e-6
        b = 5e-6
        xtip = 0.5 * width * 2e6 * a
        ytip = length * 2e6 * b - 2

        zero_width = round(0.5 * (width - tip_width))
        base_length = length - tip_length
        
        base = np.ones((width, base_length))
        zero = np.zeros((zero_width, tip_length))
        tip = np.ones((tip_width, tip_length))
        top = np.vstack((zero, tip, zero))
        topology = np.hstack((base, top))
        densities = topology
        
        super().__init__(topology, a, b, xtip, ytip, densities)
                    


class RectangularCantilever(Cantilever):
    
    def __init__(self, a, b, nelx, nely):
        
        topology = np.ones((nelx, nely))
        xtip = 2e6 * a * nelx - 2
        ytip = 2e6 * b * nely - 2
        densities = topology
        super().__init__(topology, a, b, xtip, ytip, densities)
        
        
class InitialCantileverA(Cantilever):
    
    def __init__(self):
        a = 5.0e-6
        b = 5.0e-6
        nelx = 20
        nely = 80
        xtip = 1e6 * (a * nelx)
        ytip = 1e6 * (2 * nely * b - b)
        print('-- Initial Design Space A --')
        print('Each element is %g x %g um' % (2*a*1e6, 2*b*1e6))
        print('The design area is %g x %g um' % (2*a*nelx*1e6, 2*b*nely*1e6))
        print('(xtip, ytip) = (%g, %g) um' % (self.xtip,self.ytip))
        topology = np.ones((nelx, nely))
        super().__init__(topology, a, b, xtip, ytip, topology)
        
        
class InitialCantileverB(Cantilever):
    
    def __init__(self):
        
        a = 5.0e-6
        b = 5.0e-6
        nelx = 80
        nely = 80
        xtip = 1e6 * (a * nelx)
        ytip = 1e6 * (2 * nely * b - b)
        
        topology = np.ones((nelx, nely))
        
        top = 1e-9 * np.ones((30, nely))
        mid = np.ones((20, nely))
        bot = 1e-9 * np.ones((30, nely))
        densities = np.vstack((top, mid, bot))
        
        super().__init__(topology, a, b, xtip, ytip, densities)
        
        print('-- Initial Design Space B --')
        print('Each element is %g x %g um' % (2*a*1e6, 2*b*1e6))
        print('The design area is %g x %g um' % (2*a*nelx*1e6, 2*b*nely*1e6))
        print('(xtip, ytip) = (%g, %g) um' % (self.xtip,self.ytip))