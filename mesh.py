import numpy as np


class UniformMesh(object):
    """The optimization problem is defined for a fixed mesh. The nodes 
    along the x-axis (y=0) are on a clampled boundary.
    """
    
    def __init__(self, cantilever, fem_type):
        """
        Parameters
        ----------
        cantilever : object
        An object describing a the topology of the cantilever. This requires
        a binary matrix called (topology) to describe which element in the mesh
        exists.
        
        fem_type : string
        Either 'laminate' or 'plate', depending on the material being modeled.
        """
        self.n_node = 0
        self.n_elem = 0
        self.n_mdof = 0
        self.n_edof = 1  # TODO: This is currently fixed, it shouldn't be.
        self.a = cantilever.a
        self.b = cantilever.b
        
        nelx, nely = cantilever.topology.shape
        
        self._cantilever = cantilever
        
        def row(i): return [Node(i, j, fem_type) for j in range(nely + 1)]
        self._nodes = [row(i) for i in range(nelx + 1)]

        self._elements = []
        for i, j in np.ndindex(nelx, nely):
            if cantilever.topology[i][j] == 1:
                e = self._create_element(i, j)
                self._elements.append(e)
                if cantilever.densities is not None:
                    e.set_penalty(cantilever.densities[i][j])
                
        for row in self._nodes:
            for n in row:
                self._validate_node(n)
                
        self.nelx = nelx
        self.nely = nely
    
        
    def get_finite_element_parameters(self):
        return self.a, self.b   
    
    
    def get_elements(self):
        return self._elements
    
    
    def get_node(self, i, j):
        return self._nodes[i][j]
    
    
    def get_nodes(self):
        return self._nodes
    
    
    def _validate_node(self, node):
        nn = len(node.mechanical_dof)
        if node.void == False:
            if node.boundary == False:
                for i in range(nn):
                    node.mechanical_dof[i] = self.n_mdof + i
                self.n_mdof += nn
            node.id = self.n_node
            self.n_node += 1
            
    
    def _create_element(self, i, j):
        nsw = self._nodes[i][j]
        nse = self._nodes[i + 1][j]
        nne = self._nodes[i + 1][j + 1]
        nnw = self._nodes[i][j + 1]
        e = Element(nsw, nse, nne, nnw, self.n_elem)
        for n in e.nodes:
            n.void = False
        self.n_elem += 1
        return e
    
   
    def to_console(self):
        print('-- Elements --')
        for e in self._elements:
            print(e)
        print('\n-- Nodes --')
        for row in self._nodes:
            for n in row:
                if n.void == False:
                    print(n)
                    
                    
class UniformMeshB(object):
    """The optimization problem is defined for a fixed mesh. The nodes 
    along the x-axis (y=0) are on a clampled boundary.
    """
    
    def __init__(self, cantilever, fem_type):
        """
        Parameters
        ----------
        cantilever : object
        An object describing a the topology of the cantilever. This requires
        a binary matrix called (topology) to describe which element in the mesh
        exists.
        
        fem_type : string
        Either 'laminate' or 'plate', depending on the material being modeled.
        """
        self.n_node = 0
        self.n_elem = 0
        self.n_mdof = 0
        self.n_edof = 1  # TODO: This is currently fixed, it shouldn't be.
        self.a = cantilever.a
        self.b = cantilever.b
        
        nelx, nely = cantilever.topology.shape
        
        self._cantilever = cantilever
        
        def row(i): return [Node(i, j, fem_type) for j in range(nely + 1)]
        self._nodes = [row(i) for i in range(nelx + 1)]

        self._elements = []
        for i, j in np.ndindex(nelx, nely):
            if cantilever.topology[i][j] == 1:
                e = self._create_element(i, j)
                self._elements.append(e)
                if cantilever.densities is not None:
                    e.set_penalty(cantilever.densities[i][j])
                
        for row in self._nodes:
            for n in row:
                self._validate_node(n)
                
        self.nelx = nelx
        self.nely = nely
    
        
    def get_finite_element_parameters(self):
        return self.a, self.b   
    
    
    def get_elements(self):
        return self._elements
    
    
    def get_node(self, i, j):
        return self._nodes[i][j]
    
    
    def get_nodes(self):
        return self._nodes
    
    
    def _validate_node(self, node):
        nn = len(node.mechanical_dof)
        if node.void == False:
            if node.boundary == False:
                for i in range(nn):
                    node.mechanical_dof[i] = self.n_mdof + i
                self.n_mdof += nn
            node.id = self.n_node
            self.n_node += 1
            
    
    def _create_element(self, i, j):
        nsw = self._nodes[i][j]
        nse = self._nodes[i + 1][j]
        nne = self._nodes[i + 1][j + 1]
        nnw = self._nodes[i][j + 1]
        e = Element(nsw, nse, nne, nnw, self.n_elem)
        for n in e.nodes:
            n.void = False
        self.n_elem += 1
        return e
    
   
    def to_console(self):
        print('-- Elements --')
        for e in self._elements:
            print(e)
        print('\n-- Nodes --')
        for row in self._nodes:
            for n in row:
                if n.void == False:
                    print(n)


class Element(object):
    """
    Attributes
    ----------
    self.nodes : tuple of objects 
    A four element tuple that stores the nodes of the element. The element is 
    rectangular with a node in each corner. The nodes, denoted by their position 
    on a compass, are stored in order (sw, se, ne, nw).
    """
    def __init__(self, nsw, nse, nne, nnw, elem_id):
        
        # TODO: change all coordinate to um.
        
        # Public Attributes.
        self.nodes = (nsw, nse, nne, nnw)
        self.density = 1
        self.density_penalty = 1
        self.elastic_penalty = 1
        self.piezo_penalty = 1
        self.cap_penalty = 1
        self.x0 = nsw.i + 0.5
        self.y0 = nsw.j + 0.5
        self.id = elem_id
        self.density_grad = 1
        self.elastic_grad = 3
        self.piezo_grad = 3
        self.cap_grad = 3
        
      
    def __repr__(self):
        fields = tuple([self.id] + [n.id for n in self.nodes] 
            + [self.x0, self.y0])
        repr_ = 'Element %d: %d %d %d %d (%g, %g)' % fields
        return repr_
        
    
    def get_mechanical_dof(self):
        return [d for n in self.nodes for d in n.get_mechanical_dof()]
    
    
    def get_electrical_dof(self):
        # TODO: This fixed value only allows a single cantilever wide patch.
        return [0]
        
        
    def get_mechanical_boundary(self):
        return [b for n in self.nodes for b in n.get_boundary()]
    
    
    def get_electrical_boundary(self):
        # TODO: This fixed value only allows a single cantilever wide patch.
        return [False]


    def get_displacement(self, u):
        """Displacement (u) is the system wide displacement field and this
        function extracts the components related to this element.
        """
        dof = self.get_mechanical_dof()
        boundary = self.get_mechanical_boundary()
        ue = np.zeros((len(dof), 1))
        ue[:] = u[dof]
        ue[boundary] = 0
        return ue


    def set_penalty(self, x):
        #pmin = 0.001
        self.density = x
        self.density_penalty = x ** 3
        #self.elastic_penalty = pmin + x ** 3 * (1 - pmin)    
        self.elastic_penalty = x
        self.piezo_penalty = x ** 5 
        #self.cap_penalty = x ** 3 
        self.cap_penalty = x
        self.density_grad = 3 * x ** 2
        #self.elastic_grad = 3 * x ** 2 * (1 - pmin) 
        self.elastic_grad = 1
        self.piezo_grad = 5 * x ** 4
        #self.cap_grad = 3 * x ** 2
        self.cap_grad = 1


class Node(object):
    """
    Attributes
    ----------
    boundary : bool
    A flag indicating if the node is on the clamped boundary of the cantilever.
    """
    
    def __init__(self, i, j, fem_type):
        """
        Parameters
        ----------
        fem_type : string
        Either 'laminate' or 'plate', depending on the material being modeled.
        """
        n_modf_dic = {'laminate' : 5, 'plate' : 3}
        n_mdof = n_modf_dic[fem_type]
        
        self.i = i
        self.j = j
        self.id = 0
        self.mechanical_dof = [0 for _ in range(n_mdof)]
        self.boundary = True if j == 0 else False
        self.void = True

        
    def __repr__(self):
        s1 = 'Node {0:d}: {1:d} {2:d}'.format(self.id, self.i, self.j)
        s2 = '  boundary = {0!s}'.format(self.boundary)
        return ''.join((s1, s2))
        
        
    def get_mechanical_dof(self):
        """Returns the mechanical degrees of freedom. For the laminate FEM these
        are the mid-plane x-displacement, mid-plane y-displacement, mid-plane
        z-displacement, rotation about the x-axis, and rotation about the 
        y-axis.
        """
        return self.mechanical_dof
    
    
    def get_deflection_dof(self):
        index = 0 if len(self.mechanical_dof) == 3 else 2
        return self.mechanical_dof[index]
        

    def get_boundary(self):
        """For a clamped boundary condition, all mechanical DOFs are fixed at 0.
        """
        return [self.boundary for _ in range(len(self.mechanical_dof))]
    