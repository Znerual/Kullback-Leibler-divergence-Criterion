from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='kullback_leibler_divergence_criterion',
    version=0.2,
    url='https://github.com/Znerual/Kullback-Leibler-divergence-Criterion',
    author='Laurenz Ruzicka',
    author_email='laurenz.ruzicka@tuwien.student.ac.at',
    ext_modules=cythonize('kullback_leibler_divergence_criterion.pyx'),
    include_dirs=[numpy.get_include()]
)
