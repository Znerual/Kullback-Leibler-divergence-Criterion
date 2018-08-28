from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='kullback_leibner_divergence_criterion',
    version=0.1,
    url='https://github.com/Znerual/Kullback-Leibner-divergence-Criterion',
    author='Laurenz Ruzicka',
    author_email='laurenz.ruzicka@tuwien.student.ac.at',
    ext_modules=cythonize('kullback_leibner_divergence_criterion.pyx'),
    include_dirs=[numpy.get_include()]
)
