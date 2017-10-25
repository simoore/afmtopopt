import numpy as np
import scipy.sparse as sparse
import time

ke = np.random.rand(20, 20)

n_dofs = 0
n_elem = 2000
elements = [None for _ in range(n_elem)]
for ii in range(n_elem):
    elements[ii] = list(range(n_dofs, n_dofs + 20))
    n_dofs += 20
  
    
def assemble_element_matrix(element):
    nr, nc = ke.shape
    num = nr * nc
    row = np.zeros(num)
    col = np.zeros(num)
    val = np.zeros(num)
    ntriplet = 0
    
    for ii in range(nr):
        for jj in range(nc):
            row[ntriplet] = element[ii]
            col[ntriplet] = element[jj]
            val[ntriplet] = ke[ii, jj]
            ntriplet += 1
    
    shape = (n_dofs, n_dofs)
    mat = sparse.coo_matrix((val, (row, col)), shape=shape)
    return mat
    
# Assemble element matrix and add. This is much slower with an execution time
# of 5.546554 seconds.
s1 = time.time()
matricies = [assemble_element_matrix(e) for e in elements]
total_a = sum(matricies)
e1 = time.time()
print(e1 - s1)

def assemble_matrix(elements):
    nr, nc = ke.shape
    num = nr * nc * len(elements)
    row = np.zeros(num)
    col = np.zeros(num)
    val = np.zeros(num)
    ntriplet = 0
    
    for e in elements:
        for ii in range(nr):
            for jj in range(nc):
                row[ntriplet] = e[ii]
                col[ntriplet] = e[jj]
                val[ntriplet] = ke[ii, jj]
                ntriplet += 1
    
    shape = (n_dofs, n_dofs)
    mat = sparse.coo_matrix((val, (row, col)), shape=shape)
    return mat

# This one is much quicker with execution time of 0.4450445 seconds.
s1 = time.time()
total_b = assemble_matrix(elements)
e1 = time.time()
print(e1 - s1)
