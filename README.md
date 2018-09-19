# Kullback-Leibler-divergence-Criterion

WORK-IN-PROGRESS!

The goal of this project is to offer a criterion for SciKit-Learn, which enables hypothesis testing optimised for discovery.
To reach this goal, I implement the Kullback-Leibler divergence criterion.

# Build (to get the Criterion)

The files are written and tested with Python 2.7 and Cython!

If you have installed SciKit learn from source code package, all you have to do is to compile the .pyx file with the command:

```
python setup.py build_ext --inplace
```

In case you have installed SciKit learn using pip, you are going to need the cython header file _criterion.pxd, which can be found under:

- https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/tree

Place this file in the sklearn/tree folder, which can be found in the installation directory, you still need to compile the file!

I had some problems with NumPy while compiling, under Ubuntu 18, setting the flags solved it.

```
export CFLAGS="-I /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/ $CFLAGS"
```
