import numpy as np
import abc


class Cantilever(abc.ABC):
    """Cantilevers are defined by three parameters. (topology) a binary matrix
    indicating the existence of elements in a rectangular mesh, (a) half the 
    width of the element, (b) have the length of the element. The first index
    is the x-coordinate and the second index is the y-coordinate. Elements
    along the x-axis (y == 0) are on the boundary.
    """
    
    def __init__(self, topology, a, b, xtip, ytip, densities, name):
        """xtip and ytip are in um.
        """
        self.topology = topology
        self.a = a
        self.b = b
        self.xtip = xtip
        self.ytip = ytip
        self.densities = densities
        self.name = name
        
        
    def to_console(self):
        
        nelx, nely = self.topology.shape
        dimensions = (2e6 * self.a, 2e6 * self.b)
        mesh = (2e6 * self.a * nelx, 2e6 * self.b * nely)
        tip_location = (self.xtip, self.ytip)
        
        print(''.join(('--- ', self.name, ' ---\n')))
        print('Each element is %g x %g um' % dimensions)
        print('The design area is %g x %g um' % mesh)
        print('(xtip, ytip) = (%g, %g) um\n' % tip_location)

        
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
        name = 'BJN Cantilever'
        for r, c in np.ndindex(self.nr, self.nc):
            if self.in_set(r, c):
                topology[r, c] = 1
        super().__init__(topology, a, b, xtip, ytip, densities, name)
                    
            
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
        name = 'Stepped Cantilever'
        
        super().__init__(topology, a, b, xtip, ytip, densities, name)
                    


class RectangularCantilever(Cantilever):
    
    def __init__(self, a, b, nelx, nely):
        
        topology = np.ones((nelx, nely))
        xtip = 2e6 * a * nelx - 2
        ytip = 2e6 * b * nely - 2
        densities = topology
        name = 'Rectangular Cantilever'
        super().__init__(topology, a, b, xtip, ytip, densities, name)
        
        
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
        name = 'Initial Design Space A'
        super().__init__(topology, a, b, xtip, ytip, topology, name)
        
        
class InitialCantileverB(Cantilever):
    
    def __init__(self):
        
        name = 'Initial Design Space B'
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
        
        super().__init__(topology, a, b, xtip, ytip, densities, name)

        
class InitialCantileverFixedTip(Cantilever):
    
    def __init__(self):
        
        a = 10.0e-6
        b = 10.0e-6
        nelx = 40
        nely = 40
        xtip = 1e6 * (a * nelx)
        ytip = 1e6 * (2 * nely * b - b)
        name = 'Fixed Tip Cantilever'
        
        topology = np.ones((nelx, nely))
        
        top = 1e-4 * np.ones((10, nely))
        mid = np.ones((20, nely))
        bot = 1e-4 * np.ones((10, nely))
        densities = np.vstack((top, mid, bot))
        
        super().__init__(topology, a, b, xtip, ytip, densities, name)
        
        
class InitialCantileverHigherFreq(Cantilever):
    
    def __init__(self):
        
        a = 10.0e-6
        b = 10.0e-6
        nelx = 40
        nely = 40
        xtip = 1e6 * (a * nelx)
        ytip = 1e6 * (2 * nely * b - b)
        
        topology = np.ones((nelx, nely))
        
        btop = 1e-4 * np.ones((5, 20))
        bmid = np.ones((30, 20))
        bbot = 1e-4 * np.ones((5, 20))
        base = np.vstack((btop, bmid, bbot))
        utop = 1e-4 * np.ones((15, 20))
        umid = np.ones((10, 20))
        ubot = 1e-4 * np.ones((15, 20))
        top = np.vstack((utop, umid, ubot))
        densities = np.hstack((base, top))
        
        name = 'Higher Frequency Cantilever'
        
        super().__init__(topology, a, b, xtip, ytip, densities, name)
        