# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import sys


def all_scenario_parameters():
    """
    the

    n_stock: {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
    rolling_window_size: {50, 60, ..., 240}
    alpha: (0,5, 0.55, ..., 0.95)
    setting: ("compact", "general")
    """

    n_stocks = range(5, 50+5, 5)
    window_sizes = range(50, 240+10, 10)
    percent_alphas = range(50, 100, 5)

    all_params = [(m, h, a)
                  for m in n_stocks
                  for h in window_sizes
                  for a in percent_alphas
                  ]
    # preclude Mc50_h50
    for a in percent_alphas:
        all_params.remove((50, 50, a))

    return set(all_params)
if __name__ == '__main__':
    main()