# AFM Cantilever Topology Optimizer

## Todo

* Test connectivity logic.
* Add options for fixed elements elements in the symmetry operator. This 
    involves adding a tip radius parameter to the cantilever class to indicate 
    which elements to keep around the tip.
* Change boundary conditions in connectivity operator to the tip.
* Check mode shapes of solutions the appear to have torsional solutions. Print 
    mode shapes of the solution. Maybe locally infeasible solutions have a 
    torsional mode.
    

## Numerical and Other Issues

* The dynamic range of the objective function is quite small, and as such, 
    objective function scaling is required. The ultimate solution to this 
    problem is to properly scale the finite element analysis. Several Ipopt 
    options for the restoration phase alleviate these issues.
* Note that the Ipopt gradient checker doesn't use the initial values. This 
    plays havoc with the structural analysis and causes inconsistent jacobian 
    values calculated with finite differences.
* The selection of the objective scaling is performed as follows. With the 
    automatric gradient based scaling option, a set of identical problems with 
    various scaling factors was executed. When a cluster of solutions is 
    approximately the same, a scaing factor in the center of that cluster was 
    employed. Larger scaling tends to find better solutions. Choosing 1e9. 
    Locally infeasible solutions may be due to torisional modes.
* The execution of the Ipopt routines catches exceptions and when the return 
    data is properly initialized, execution continues with the incorrect 
    numbers. Unit testing is required.


## Installation of Ipopt on Windows

The [Ipopt](https://projects.coin-or.org/Ipopt) library performs the 
optimization and 
[cyipopt](https://github.com/matthias-k/cyipopt) provides a python wrapper. 
For this work, the 64 bit version 3.11.0 
[precompiled binaries](https://www.coin-or.org/download/binary/Ipopt) 
(Ipopt-3.11.0-Win32-Win64-dll.7z) were used.

The cyipopt documents a number of dependancies. Here I'll quickly document the 
installation of two of them:

* The cython compiler: 

```
pip install cython
```

* [Visual C++ Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools).


After the installation of the dependancies, download the cyipopt and unzip the 
python package. Copy the contents of the precompiled binaries, that is a `lib` 
and an `include` directory, into the cyipopt directory. Then within 
cyipopt's `setup.py` modify the definitions of the following variables:

```
IPOPT_ICLUDE_DIRS = ['include/coin', np.get_include()]
IPOPT_LIBS = ['IpOpt-vc10', 'IpOptFSS']
IPOPT_LIB_DIRS = ['lib/x64/ReleaseMKL']
IPOPT_DLL = ['IpOpt-vc10.dll', 'IpOptFSS.dll']
```

To instal the library, execute the setup script in the cyipopt directory:

```
python setup.py install
```

If you have a few errors associated with importing python modules in the 
installed library, the proposed quick fix is suggested. In the file 
`ipopt/__init__.py` change:

```
-from cyipopt import *
+from .cyipopt import *
```

and in the file `ipopt/ipopt_wrapper.py` change:

```
-import cyipopt
+import ipopt.cyipopt as cyipopt
```

The reexecute the `setup.py` script.


## Extensions

* Extend to higher order modes: add polarity of piezoelectric material for 
    higher order modes.
* Add a delta filter for tip displacement for non-changing tip location.
* Allow for more than one electrical DOF.
* Make the tip location an optimization design parameter.
