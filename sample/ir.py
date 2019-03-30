# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import sys
import numpy as np

def main():
    size = 5
    weights = np.arange(6, 1, -1)
    experts = np.tile(weights, (size * (size-1), 1))
    print(experts)
    row = 0
    for idx in range(size):
        for jdx in range(size):
            if idx != jdx:
                print(idx, jdx, row)
                experts[row, jdx] += experts[row, idx]
                experts[row, idx] = 0
                row += 1
    print(experts)





if __name__ == '__main__':
    main()