# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import os
import platform
import glob
import time

import zmq
import portfolio_programming as pp

def get_zmq_version():
    node = platform.node()
    print("Node:{} libzmq version is {}".format(node, zmq.zmq_version()))
    print("Node:{} pyzmq version is {}".format(node, zmq.__version__))


def _all_spsp_cvar_params(setting):
    """
    "report_SPSP_CVaR_{}_scenario-set-idx{}_{}_{}_M{}_Mc{}_h{}_a{:.2f}_s{
    }.pkl".format(
                self.setting,
                self.scenario_set_idx,
                self.exp_start_date.strftime("%Y%m%d"),
                self.exp_end_date.strftime("%Y%m%d"),
                self.max_portfolio_size,
                self.n_symbol,
                self.rolling_window_size,
                self.alpha,
                self.n_scenario
            )
    """
    REPORT_FORMAT = "repot_SPSP_CVaR_{setting}_scenario-set-idx{sdx}_{" \
                    "exp_start_date}_{exp_end_date}_M{max_portfolio}_Mc{" \
                    "n_candidate_symbol}_h{rolling_window_size" \
                    "}_a{alpha}_s{n_scenario}.pkl"
    if setting not in ('compact', 'general'):
        raise ValueError('Wrong setting: {}'.format(setting))

    # set_indices = (1, 2, 3)
    set_indices = (1,)
    s_date = pp.SCENARIO_START_DATE.strftime("%Y%m%d")
    e_date = pp.SCENARIO_END_DATE.strftime("%Y%m%d")
    max_portfolio_sizes = range(5, 50 + 5, 5)
    window_sizes = range(60, 240 + 10, 10)
    n_scenarios = [200, ]
    alphas = ["{:.2f}".format(v / 100.) for v in range(50, 100, 10)]

    # dict comprehension
    # key: file_name, value: parameters
    if setting == "compact":
        return {
            REPORT_FORMAT.format(
                setting=setting,
                sdx=sdx,
                exp_start_date=s_date,
                exp_end_date=e_date,
                max_portfolio=m,
                n_candidate_symbol=m,
                rolling_window_size=h,
                alpha=a,
                n_scenario=s
            ): (setting, sdx, s_date, e_date, m, h, float(a), s)
            for sdx in set_indices
            for m in max_portfolio_sizes
            for h in window_sizes
            for a in alphas
            for s in n_scenarios
        }

    elif setting == "general":
        return {
            REPORT_FORMAT.format(
                setting=setting,
                sdx=sdx,
                exp_start_date=s_date,
                exp_end_date=e_date,
                max_portfolio=m,
                n_candidate_symbol=50,
                rolling_window_size=h,
                alpha=a,
                n_scenario=s
            ): (setting, sdx, s_date, e_date, m, h, float(a), s)
            for sdx in set_indices
            for m in max_portfolio_sizes
            for h in window_sizes
            for a in alphas
            for s in n_scenarios
        }


def checking_existed_spsp_cvar_report(setting, report_dir=None):
    """
    return unfinished experiment parameters.
    """
    if report_dir is None:
        report_dir = pp.REPORT_DIR
    all_reports = _all_spsp_cvar_params(setting)

    os.chdir(report_dir)
    existed_reports = glob.glob("*.pkl")
    for report in existed_reports:
        all_reports.pop(report, None)

    # unfinished params
    return all_reports


def parameters_server(setting="compact"):
    node = platform.node()
    pid = os.getpid()
    context = zmq.Context()

    # zmq.sugar.socket.Socket
    socket = context.socket(zmq.REP)

    # Protocols supported include tcp, udp, pgm, epgm, inproc and ipc.
    socket.bind("tcp://*:25555")

    params = checking_existed_spsp_cvar_report(setting).values()
    workers = {}
    while len(params):
        # Wait for request from client
        client_node_pid = socket.recv_string()
        print("Received request: {}".format(client_node_pid))
        node, pid = client_node_pid.split('_')
        workers.setdefault(node, 0)
        workers[node] += 1

        #  Send reply back to client
        work = params.pop()
        print("send {} to {}".format(work, client_node_pid))
        socket.send_pyobj(params.pop())

    socket.close()
    context.term()


def parameter_client():
    node = platform.node()
    pid = os.getpid()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    url = "tcp://localhost:25555"
    socket.connect(url)

    # for IO monitoring
    poll = zmq.Poller()
    poll.register(socket, zmq.POLLIN)

    node_pid = "{}_{}".format(node, pid)
    while True:
        # send request to server
        socket.send_string( node_pid)
        socks = dict(poll.poll(10000))

        if socks.get(socket) == zmq.POLLIN:
            # still connected
            # receive parameters from server
            work = socket.recv_pyobj()
            print(work)
            time.sleep(3)
            # break
        else:
            # no response from server, reconnected
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
            poll.unregister(socket)

            # reconnection
            socket = context.socket(zmq.REQ)
            socket.connect(url)
            poll.register(socket, zmq.POLLIN)

            socket.send_string(node_pid)
            print('reconnect to {}'.format(url))

    socket.close()
    context.term()


if __name__ == '__main__':
    get_zmq_version()

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--server", default=False,
                        action='store_true',
                        help="parameter server mode")

    parser.add_argument("-c", "--client", default=True,
                        action='store_true',
                        help="run SPSP_CVaR client mode")

    args = parser.parse_args()
    if args.server:
        print("run SPSP_CVaR parameter server mode")
        parameters_server()
    elif args.client:
        print("run SPSP_CVaR client mode")
        parameter_client()
    else:
        raise ValueError("no mode is set.")
