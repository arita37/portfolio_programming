# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import zmq

def get_zmq_version():
    print("Current libzmq version is {}".format(zmq.zmq_version()))
    print("Current  pyzmq version is {}".format(zmq.__version__))

if __name__ == '__main__':
    get_zmq_version()

