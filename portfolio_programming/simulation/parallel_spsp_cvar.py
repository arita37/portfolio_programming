# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import zmq

def get_zmq_version():
    print("Current libzmq version is {}".format(zmq.zmq_version()))
    print("Current  pyzmq version is {}".format(zmq.__version__))


def parameters_pool():
    context = zmq.Context()

    # zmq.sugar.socket.Socket
    socket = context.socket(zmq.REP)

    # Protocols supported include tcp, udp, pgm, epgm, inproc and ipc.
    socket.bind("tcp://*:25555")

    params = range(10)
    while len(params):
        # Wait for next request from client
         message = socket.recv()
         print("Received request: %s" % message)

         #  Do some 'work'
         time.sleep(1)

         #  Send reply back to client
         socket.send_string("World")


if __name__ == '__main__':
    get_zmq_version()

