# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import numpy as np



def pump_failure():
    times = np.array([94, 16, 63, 126, 5, 31, 1, 1, 2, 10])
    n_failures = np.array([5, 1, 5, 14, 3, 19, 1, 1, 4, 22])
    data = np.vstack((times, n_failures))
    print(data)


if __name__ == '__main__':
    pump_failure()
