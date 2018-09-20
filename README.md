# Kullback-Leibler-divergence-Criterion

The goal of this project is to offer a criterion for SciKit-Learn, which enables hypothesis testing optimised for discovery.
To reach this goal, I implement the Kullback-Leibler divergence criterion.

# Build (to get the Criterion)
The only files you need to implement the kullback-Leibler divergence criterion are the setup.py and the kullback_leibler_divergence_criterion.pyx from this repository (I'll get to the header later on), all the other files are used to generate plots and manipulate data.

The files are written and tested with Python 2.7 and Cython!

If you have installed SciKit learn from source code package, all one has to do is to compile the .pyx file with the command:

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
# How to use the examples files
For the examples to work, you need the correct ROOT data file, which needs to be read out by the ttz_dataset.py script. This script generates a .h5 file, which than will be used for all the other scripts as input. 
You also need to change the directory imports, which consist of paths to an output directory (for the plots) and a store directory (for the data). You can either generate a user.py file under TTXPheno/Tools and define the paths there, or manually change them in code.
Others things to keep in mind, one needs https://github.com/TTXPheno to use the logger and the plot methods in ttz_dataset.py (also for the directories)

#Examples
You can find a pre-processed .h5 file and plots under:
http://www.hephy.at/xuser/lruzicka/
