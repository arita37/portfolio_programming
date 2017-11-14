# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import platform
import os

node_name = platform.node()

if node_name == 'X220':
    # windows 10
    PROJECT_DIR = r'C:\Users\chen1\Documents\workspace_pycharm\portfolio_programming'
    TMP_DIR = r'e:'
else:
    # ubuntu linux 16.04
    PROJECT_DIR = r'/home/chenhh/workspace_pycharm/portfolio_programming'
    TMP_DIR = r'/tmp'

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
