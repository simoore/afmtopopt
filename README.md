## Todo

* Extend to higher order modes: add polarity of piezoelectric material for higher order modes.
* Vectorize more of the finite element analysis.
* Check sensitivies for large cantilever similiar to _analysis_for_sensitivities. There also appears
    to be some convergence issues, it tends to oscillate around the final solution. This appears at the solution
    Algorithm converged to a point of local infeasibility. Problem may be infeasible.
* Still producing poorly connected structures... Propose a piezoelectric filter to penalize elements to 
    properly connected to the structure.
* Add options for fixed elements elements in the symmetry operator. This involves adding a tip radius parameter
    to the cantilever class to indicate which elements to keep around the tip.
* Add a delta filter for tip displacement for non-changing tip location.
* Add more elegant composition of regularization operators.


## Installation of Ipopt on Windows

The [Ipopt](https://projects.coin-or.org/Ipopt) library performs the optimization and 
[cyipopt](https://github.com/matthias-k/cyipopt) provides a python wrapper. For this work, the 64 bit version 3.11.0 
[precompiled binaries](https://www.coin-or.org/download/binary/Ipopt) (Ipopt-3.11.0-Win32-Win64-dll.7z) were used.

The cyipopt documents a number of dependancies. Here I'll quickly document the installation of two of them:

* The cython compiler: 

```
pip install cython
```

* [Visual C++ Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools).


After the installation of the dependancies, download the cyipopt and unzip the python package. Copy the contents 
of the precompiled binaries, that is a `lib` and an `include` directory, into the cyipopt directory. Then within 
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

If you have a few errors associated with importing python modules in the installed library, the proposed 
quick fix is suggested. In the file `ipopt/__init__.py` change:

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
