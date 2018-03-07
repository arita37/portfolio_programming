# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import datetime as dt
import sys
from time import sleep
import ipyparallel as ipp


def parallel_hello():
    rc = ipp.Client(profile='ssh')
    dv = rc[:]
    dv.use_dill()
    dv.scatter('engine_id', rc.ids, flatten=True)
    print("Engine IDs: ", dv['engine_id'])
    n_engine = len(rc.ids)

    with dv.sync_imports():
        import portfolio_programming.simulation.hello

    lbv = rc.load_balanced_view()
    print("start map unfinished parameters to load balance view.")
    ar = lbv.map_async(
        portfolio_programming.simulation.hello.hello,
        range(60))

    while not ar.ready():
        print("{} n_engine:{} j hello task: {}/{} {:10.1f} "
              "secs".format(
            str(dt.datetime.now()), n_engine, ar.progress, len(ar),
            ar.elapsed))
        sys.stdout.flush()
        sleep(10)


if __name__ == '__main__':
    parallel_hello()
