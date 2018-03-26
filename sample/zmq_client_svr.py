# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
test of zeromq client and server architecture
"""

import datetime as dt
import multiprocessing as mp
import os
import platform
from time import (time, sleep)

import zmq


def task_server(task_parameters, port=34567):
    """
    Parameters
    ---------------
    task_parameters : list or iterable object
    port : positive integer
    """
    node = platform.node()
    pid = os.getpid()
    server_node_pid = "{}[pid:{}]".format(node, pid)
    context = zmq.Context()

    # zmq.sugar.socket.Socket
    task_socket = context.socket(zmq.REP)
    cmd_socket = context.socket(zmq.PUB)

    # Protocols supported include tcp, udp, pgm, epgm, inproc and ipc.
    task_socket.bind("tcp://*:{}".format(port))
    cmd_socket.bind("tcp://*:{}".format(port+10))

    # multiprocessing queue is thread-safe.
    tasks = mp.Queue()
    [tasks.put(v) for v in task_parameters]
    client_node_pid_set = set()
    client_node_count = {}
    print("Ready to serving, remaining {} parameters.".format(tasks.qsize()))
    svr_start_time = dt.datetime.now()
    t0 = time()

    while not tasks.empty():
        # Wait for request from client
        param = tasks.get()
        task_socket.send_pyobj(param)
        cmd_socket.send_pyobj("running")

        task_res = task_socket.recv_pyobj()
        task_stat = cmd_socket.recv_pyobj()
        print("recv:", task_res)
        print("task_stat:", task_stat)

    cmd_socket.send_pyobj("empty")

    print("end of serving, remaining {} parameters.".format(tasks.qsize()))

    task_socket.close()
    cmd_socket.close()
    context.term()
    tasks.close()


def task_client(task_func, server_ip="140.117.168.49", server_port=34567):
    """
    Parameters
    ------------------------
    task_func : function object
    server_ip : string
    server_port : positive integer
    """
    node = platform.node()
    pid = os.getpid()

    context = zmq.Context()
    task_url = "tcp://{}:{}".format(server_ip, server_port)
    cmd_url = "tcp://{}:{}".format(server_ip, server_port + 10)
    task_socket = context.socket(zmq.REQ)
    cmd_socket = context.socket(zmq.SUB)
    task_socket.connect(task_url)
    cmd_socket.connect(cmd_url)

    #  monitor socket input data
    poll = zmq.Poller()
    poll.register(task_socket, zmq.POLLIN)
    poll.register(cmd_socket, zmq.POLLIN)

    interrupt = False
    reconnect_count = 0
    while not interrupt:
        # send request to server
        socks = dict(poll.poll(2000))
        print("task in socks:", task_socket in socks)
        print("cmd in socks:", cmd_socket in socks)

        if task_socket in socks and socks[task_socket] == zmq.POLLIN:
            # data from task socket, still connected
            # receive parameters from server
            task_param = task_socket.recv_pyobj()
            print("{:<15} recv: {}".format(
                str(dt.datetime.now()),
                task_param))
            try:
                res = task_func(*task_param)
                task_socket.send_pyobj(res)
            except Exception as e:
            # no response from server, reconnected
            reconnect_count += 1
            if reconnect_count >= 5:
                break

            # manual close socket
            for sock in (task_socket, cmd_socket):
                sock.setsockopt(zmq.LINGER, 0)
                sock.close()
                poll.unregister(sock)

            # reconnection
            task_socket = context.socket(zmq.REQ)
            cmd_socket = context.socket(zmq.SUB)
            task_socket.connect(task_url)
            cmd_socket.connect(cmd_url)
            poll.register(task_socket, zmq.POLLIN)
            poll.register(cmd_socket, zmq.POLLIN)

            # socket.send_string(node_pid)
            print('reconnect to {}'.format(task_url))

        if cmd_socket in socks and socks[cmd_socket] == zmq.POLLIN:
            # message from cmd socket
            msg = cmd_socket.recv_pyobj()
            print(msg)
            if msg == 'empty':
                task_empty = True

    print("all task done.")
    task_socket.close()
    cmd_socket.close()
    context.term()


def my_func(*args):
    print("my_func:{}".format(args))
    if args[1] % 3 == 0:
        raise ValueError('even number.')
    sleep(2)
    return 1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", default=False,
                        action='store_true',
                        help="task server mode")

    parser.add_argument("-c", "--client", default=False,
                        action='store_true',
                        help="task client mode")
    args = parser.parse_args()
    if args.server:
        print("run task server mode")
        task_server([[i, i + 1] for i in range(10)])
    elif args.client:
        print("run task client mode")
        task_client(my_func, server_ip="localhost", server_port=34567)
