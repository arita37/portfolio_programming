# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3

https://github.com/cython/cython/wiki/InstallingOnWindows
https://github.com/cython/cython/wiki/CythonExtensionsOnWindows

You can then "cd CYTHON_FILE_PATH" to navigate to your Python app and then
build your C extensions by entering:
     python setup.py build_ext --inplace --compiler=msvc

the MSVC compiler default optimization level: /Ox (Full Optimization)
https://msdn.microsoft.com/en-us/library/59a3b321.aspx
"""

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
              "spsp_cvar.pyx",
              ["spsp_cvar.pyx",],
              include_dirs = [np.get_include()],
    ),
]

setup(
      name = 'portfolio_programming',
      author = 'Hung-Hsin Chen',
      author_email = 'chenhh@par.cse.nsysu.edu.tw',
      ext_modules = cythonize(extensions),
) 
