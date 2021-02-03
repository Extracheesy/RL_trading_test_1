# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:42:24 2021

@author: despo
"""

from cx_Freeze import setup, Executable

setup(name = "run_MainTest", version = "0.1",
      description = "Run Data version for MlpLstmPolicy and MlpLnLstmPolicy",
      executables = [Executable("run_MainTest.py")])