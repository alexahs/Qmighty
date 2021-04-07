#!/usr/bin/env python

from distutils.core import setup

setup(name="qmighty",
      version="0.1",
      description="Tool for queueing and monitoring LAMMPS simulations",
      author="Alexander Sexton",
      author_email="alexahs@uio.no",
      url="github.com/alexahs/qmighty",
      install_requires = ["lammps_logfile"],
      scripts=["qmighty.py"]
)
