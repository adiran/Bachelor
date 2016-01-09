from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'functions',
  ext_modules = cythonize("functions.pyx"),
  include_dirs=[numpy.get_include()]
)

setup(
  name = 'qualitycheck',
  ext_modules = cythonize("qualitycheck.pyx"),
  include_dirs=[numpy.get_include()]
)

setup(
  name = 'listenC',
  ext_modules = cythonize("listenC.pyx"),
  include_dirs=[numpy.get_include()]
)