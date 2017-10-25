from math import ceil, sqrt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

# Python translation of an 88 line topology optimization matlab script.
#def top88(nelx, nely, volfrac, penal, rmin, ft):
nelx = 3
nely = 4
volfrac = 0.5
penal = 3
rmin = 1.5
ft = 2

# Material properties.
E0 = 1
Emin = 1e-9
nu = 0.3

# Prepare finite element analysis.
num_elem = nelx * nely
num_node = (nelx + 1) * (nely + 1)
num_dofs = 2 * num_node

A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], 
                [-6, 3, 12, -3], [-3, 0, -3, 12]])
A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6],
                [0, -3, -6, 3], [3, -6, 3, -6]])
B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], 
                [-2, -9, -4, -3], [9, 4, -3, -4]])
B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], 
                [4, 9, 2, 3], [-9, -2, 3, 2]])
A = np.vstack((np.hstack((A11, A12)), np.hstack((A12.T, A11))))
B = np.vstack((np.hstack((B11, B12)), np.hstack((B12.T, B11))))

# KE is the element stiffness with elasticity E=E0.
KE = 1/(1 - nu**2)/24 * (A + nu * B)

# nodenrs is a grid labeling each node in their proper positions. This 
# helps with the construction of the matrix edofMat. Each row of edofMat is 
# for each element. Each number in the row represents a degree of 
# freedom for that element.
nodenrs = np.reshape(range(num_node), (1+nely, 1+nelx), order='F')
edofVec = np.reshape(2*nodenrs[1:, 0:-1], num_elem, 1)
dofadj = np.array([0, 1] + [2*nely+x for x in [2, 3, 0, 1]] + [-2, -1])
edofMat = np.tile(edofVec, (8, 1)).T + np.tile(dofadj, (num_elem, 1))

# Sparse matrices are employed for the system matrices. For a fixed 
# structure, the index that each term in the matrices in known beforehand 
# for all dofs and is generated with this code.
iK = np.reshape(np.kron(edofMat, np.ones((8,1))).T, 64*num_elem, 1)
jK = np.reshape(np.kron(edofMat, np.ones((1,8))).T, 64*num_elem, 1)

# Define the point load under which the compliance is tested.

# Is a point load on a single element of the beam. Therefore it only has
# one entry.
shape = (num_dofs, num_dofs)
F = sparse.coo_matrix(([-1], ([1], [0])), shape=shape).tocsc()

# U is an unknown displacement that is solved to calculate the cost
# function. This line pre-allocates the memory. There is one value for each
# DOF.
U = np.zeros((2 * num_node, 1))

# Fixed DOFs are from nodes on the fixed boundaries. For each node there
# are two degrees of freedom, the horizontal and vertical displacement. The
# left hand edge is prevented from moving horizontally and the lower-right
# corner is prevented from moving vertically. all dofs are all DOFs and
# freedofs is alldofs minus fixedofs.
fixeddofs = np.union1d(range(0, 2*(nely+1), 2), [2*num_node-1])
alldofs = np.array(range(2 * num_node))
freedofs = np.setdiff1d(alldofs, fixeddofs)

# Prepare filter.
iH = np.zeros(num_elem * (2 * (ceil(rmin) - 1) + 1)**2)
jH = np.zeros(iH.shape[0])
sH = np.zeros(iH.shape[0])
k = 0

for i1 in range(nelx):
    for j1 in range(nely):
        e1 = i1 * nely + j1
        i2_str = max(i1 - (ceil(rmin)-1), 0)
        i2_end = min(i1 + ceil(rmin), nelx)
        for i2 in range(i2_str, i2_end):
            j2_str = max(j1 - (ceil(rmin)-1), 0)
            j2_end = min(j1 + ceil(rmin), nely)
            for j2 in range(j2_str, j2_end):
                e2 = i2 * nely + j2
                iH[k] = e1
                jH[k] = e2
                sH[k] = max(0, rmin - sqrt((i1-i2)**2+(j1-j2)**2))
                k += 1

H = sparse.coo_matrix((sH, (iH, jH))).tocsc()
Hs = np.sum(H, 1)

# Initialize iteration.
# x is the original density field, xPhys is the filtered density field.
x = np.tile(volfrac, (nely, nelx))
xPhys = x.copy()
loop = 0
change = 1

# Start iteration.
while change > 0.01:
    loop = loop + 1
    
    # FE Analysis. Scales KE for each element as a function of the elements
    # density and forms the the global stiffness matrix. sK are the entries 
    # based on the physical densities and the penalization factor. K is the 
    # complete sparse system stiffness matrix. Then U is solved for the 
    # given force vector. Also the fixed dofs are removed.
    flat_e = np.ravel(Emin + xPhys.T ** penal*(E0-Emin))[np.newaxis]
    flat_ke = np.ravel(KE)[np.newaxis]
    flat_k = flat_ke.T @ flat_e
    sK = np.reshape(flat_k, 64*num_elem, 1)
    K = sparse.coo_matrix((sK, (iK, jK)))
    K = 0.5 * (K + K.T)  # enforces symmetry
    K_freedofs = K[freedofs[:, None], freedofs]
    F_freedofs = F[freedofs]
    U_freedofs = linalg.spsolve(K_freedofs.tocsc(), F_freedofs.tocsc())
    U[freedofs] = U_freedofs[:, 0].toarray()
    
    
    # Objective function and sensitivity analysis.
    # ce is the compliance per element, it used the unmodified stiffness 
    # vector, #while using the displacement of the modified structure. This 
    # is for computational performance. c is the compliance - ie what is
    # being minimised and this is density penalities for the stiffness 
    # matrix are applied again. dc is the sensitivity of sensitivities with 
    # respect to the densities, dv is the sensivity of the volume with 
    # respect to the densities.
    u_elems = np.squeeze(U[edofMat])
    ce_flat = np.sum((u_elems @ KE) * u_elems, 1)
    ce = np.reshape(ce_flat, (nelx, nely)).T
    c = np.sum((Emin + xPhys ** penal * (E0-Emin)) * ce)
    dc = -penal * (E0-Emin) * xPhys ** (penal-1) * ce
    dv = np.ones((nely, nelx))

    # FILTERING/MODIFICATION OF SENSITIVITIES
    flat_dc = np.ravel(dc, order='F')
    if ft == 1:
        flat_x = np.ravel(x, order='F')
        x_sat = np.maximum(1e-3 * np.ones(flat_x.shape), flat_x)
        flat_dc = (H @ (flat_x * flat_dc)) / Hs.T / x_sat
    elif ft == 2:
        flat_dv = np.ravel(dv, order='F')
        flat_dc = H @ (flat_dc / Hs.T).T
        flat_dv = H @ (flat_dv / Hs.T).T
        dv = np.reshape(flat_dv, tuple(reversed(dv.shape))).T
    dc = np.reshape(flat_dc, tuple(reversed(dc.shape))).T
    dc = np.array(dc)
    dv = np.array(dv)
    
    # OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    l1, l2, move = 0, 1e9, 0.2
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew = np.minimum(x + move, x * np.sqrt(-dc/dv/lmid))
        xnew = np.clip(xnew, a_min=None, a_max=1)
        xnew = np.maximum(x - move, xnew)
        xnew = np.clip(xnew, a_min=0, a_max=None) 
        if ft == 1:
            xPhys = xnew
        elif ft == 2:
            xnew_flat = np.ravel(xnew, order='F')[np.newaxis].T
            xPhys_flat = ((H @ xnew_flat) / Hs).A
            xPhys = np.reshape(xPhys_flat, xPhys.shape, order='F')

        if np.sum(xPhys) > volfrac * num_elem:
            l1 = lmid 
        else:
            l2 = lmid

    change = np.amax(np.abs(xnew - x))
    x = xnew

    # Print results of each iteration.
    data = (loop, c, np.mean(xPhys), change)
    print('It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f' % data)
          