# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import datetime as dt
import glob
import logging
import multiprocessing as mp
import os
import pickle
import platform
import sys

import numpy as np
import xarray as xr
import zmq

import portfolio_programming as pp
from portfolio_programming.simulation.run_spsp_cvar import run_SPSP_CVaR


def get_zmq_version():
    node = platform.node()
    print("Node:{} libzmq version is {}".format(node, zmq.zmq_version()))
    print("Node:{} pyzmq version is {}".format(node, zmq.__version__))


def _all_spsp_cvar_params(setting, yearly=False):
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
    REPORT_FORMAT = "report_SPSP_CVaR_{setting}_scenario-set-idx{sdx}_{" \
                    "exp_start_date}_{exp_end_date}_M{max_portfolio}_Mc{" \
                    "n_candidate_symbol}_h{rolling_window_size" \
                    "}_a{alpha}_s{n_scenario}.pkl"
    if setting not in ('compact', 'general'):
        raise ValueError('Wrong setting: {}'.format(setting))

    # set_indices = (1, 2, 3)
    set_indices = (1, )
    
    if not yearly:
        # whole interval
        years = [(pp.SCENARIO_START_DATE.strftime("%Y%m%d"),
                  pp.SCENARIO_END_DATE.strftime("%Y%m%d"))
                 ]
    else:
        # yearly interval
        years = [(dt.date(2005, 1, 3), dt.date(2005, 12, 30)),
                 (dt.date(2006, 1, 2), dt.date(2006, 12, 29)),
                 (dt.date(2007, 1, 2), dt.date(2007, 12, 31)),
                 (dt.date(2008, 1, 2), dt.date(2008, 12, 31)),
                 (dt.date(2009, 1, 5), dt.date(2009, 12, 31)),
                 (dt.date(2010, 1, 4), dt.date(2010, 12, 31)),
                 (dt.date(2011, 1, 3), dt.date(2011, 12, 30)),
                 (dt.date(2012, 1, 2), dt.date(2012, 12, 28)),
                 (dt.date(2013, 1, 2), dt.date(2013, 12, 31)),
                 (dt.date(2014, 1, 2), dt.date(2014, 12, 31)),
                 (dt.date(2015, 1, 5), dt.date(2015, 12, 31)),
                 (dt.date(2016, 1, 4), dt.date(2016, 12, 30)),
                 (dt.date(2017, 1, 3), dt.date(2017, 12, 29))
                 ]
    max_portfolio_sizes = range(5, 50 + 5, 5)
    window_sizes = range(60, 240 + 10, 10)
    n_scenarios = [200, ]
    alphas = ["{:.2f}".format(v / 100.) for v in range(50, 100, 5)]

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
            for s_date, e_date in years
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
            for s_date, e_date in years
            for m in max_portfolio_sizes
            for h in window_sizes
            for a in alphas
            for s in n_scenarios
        }


def checking_existed_spsp_cvar_report(setting, yearly, report_dir=None):
    """
    return unfinished experiment parameters.
    """
    if report_dir is None:
        report_dir = pp.REPORT_DIR
    all_reports = _all_spsp_cvar_params(setting, yearly)

    os.chdir(report_dir)
    existed_reports = glob.glob("*.pkl")
    for report in existed_reports:
        all_reports.pop(report, None)

    # unfinished params
    return all_reports


def parameter_server(setting, yearly):
    node = platform.node()
    pid = os.getpid()
    server_node_pid = "{}[pid:{}]".format(node, pid)
    context = zmq.Context()

    # zmq.sugar.socket.Socket
    socket = context.socket(zmq.REP)

    # Protocols supported include tcp, udp, pgm, epgm, inproc and ipc.
    socket.bind("tcp://*:25555")

    # multiprocessing queue is thread-safe.
    params = mp.Queue()
    [params.put(v) for v in
     checking_existed_spsp_cvar_report(setting, yearly).values()]
    progress_node_pid = set()
    progress_node_count = {}
    finished = {}
    print("Ready to serving, remaining {} parameters.".format(params.qsize()))

    while not params.empty():
        # Wait for request from client
        client_node_pid = socket.recv_string()
        print("{:<15}, {} Received request: {}".format(
            str(dt.datetime.now()),
            server_node_pid,
            client_node_pid))

        #  Send reply back to client
        work = params.get()
        print("send {} to {}".format(work, client_node_pid))
        socket.send_pyobj(work)

        c_node, c_pid = client_node_pid.split('_')
        finished.setdefault(c_node, 0)
        if client_node_pid in progress_node_pid:
            # the node have done a work
            finished[c_node] += 1
        else:
            progress_node_count.setdefault(
                c_node, {"req_time": dt.datetime.now(), "cnt": 0})
            progress_node_count[c_node]['req_time'] = dt.datetime.now()
            progress_node_count[c_node]['cnt'] += 1

        # the progress set is not robust, because we don't track
        # if a process on a node is crashed or not.
        progress_node_pid.add(client_node_pid)

        print("remaining parameters:{}".format(params.qsize()))
        print("progressing: {}".format(len(progress_node_pid)))
        for w_node, cnt in finished.items():
            print("node:{:<8} progress:{:>3} ,finish:{:>3} last req:{}".format(
                w_node, progress_node_count[w_node]['cnt'], cnt,
                progress_node_count[w_node]['cnt']['req_time'].strftime(
                    "%Y%md%d-%H%M%S"))
            )

    print("end of serving, remaining {} parameters.".format(params.qsize()))
    socket.close()
    context.term()
    params.close()


def parameter_client(server_ip="140.117.168.49"):
    node = platform.node()
    pid = os.getpid()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    url = "tcp://{}:25555".format(server_ip)
    socket.connect(url)

    # for IO monitoring
    poll = zmq.Poller()
    poll.register(socket, zmq.POLLIN)

    node_pid = "{}_{}".format(node, pid)
    while True:
        # send request to server
        socket.send_string(node_pid)
        socks = dict(poll.poll(10000))

        if socks.get(socket) == zmq.POLLIN:
            # still connected
            # receive parameters from server
            work = socket.recv_pyobj()
            print("{:<15} receiving: {}".format(
                str(dt.datetime.now()),
                work))
            run_SPSP_CVaR(*work)

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


def aggregating_reports(setting):
    if setting not in ("compact", "general"):
        raise ValueError("Unknown SPSP_CVaR setting: {}".format(setting))

    s_date = pp.SCENARIO_START_DATE.strftime("%Y%m%d")
    e_date = pp.SCENARIO_END_DATE.strftime("%Y%m%d")
    set_indices = [1, 2, 3]
    max_portfolio_sizes = range(5, 50 + 5, 5)
    window_sizes = range(60, 240 + 10, 10)
    n_scenarios = [200, ]
    alphas = ["{:.2f}".format(v / 100.) for v in range(50, 100, 5)]

    attributes = [
        'initial_wealth', 'final_wealth',
        'cum_roi', 'daily_roi', 'daily_mean_roi',
        'daily_std_roi', 'daily_skew_roi', 'daily_ex-kurt_roi',
        'Sharpe', 'Sortino_full', 'Sortino_partial'
    ]
    report_xarr = xr.DataArray(
        np.zeros((
            len(set_indices), len(max_portfolio_sizes), len(window_sizes),
            len(alphas), len(attributes))),
        dims=("scenario_set_idx", "max_portfolio_size", "rolling_window_size",
              "alpha", "attribute"),
        coords=(set_indices, max_portfolio_sizes, window_sizes, alphas,
                attributes)
    )

    # key: report_name, value: parameters
    report_dict = _all_spsp_cvar_params(setting)
    report_count = 0
    no_report_count = 0
    for name, param in report_dict.items():
        path = os.path.join(pp.REPORT_DIR, name)
        setting, sdx, s_date, e_date, m, h, a, s = param
        alpha = "{:.2f}".format(a)
        try:
            with open(path, 'rb') as fin:
                report = pickle.load(fin)
                report_count += 1
                print("{} {:.2%}".format(report['simulation_name'],
                                         report['cum_roi']))
                for attr in attributes:
                    report_xarr.loc[sdx, m, h, alpha, attr] = report[attr]
        except FileNotFoundError as e:
            no_report_count += 1
            continue

    print("report count:{}, no report count:{}".format(
        report_count, no_report_count))

    report_xarr_path = os.path.join(pp.DATA_DIR,
                                    "report_SPSP_CVaR_{}_{}_{}.nc".format(
                                        setting, s_date, e_date))
    report_xarr.to_netcdf(report_xarr_path)


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(filename)15s %(levelname)10s %(asctime)s\n'
               '%(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        level=logging.INFO)
    import argparse

    get_zmq_version()

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--server", default=False,
                        action='store_true',
                        help="parameter server mode")

    parser.add_argument("--setting", type=str,
                        choices=("compact", "general"),
                        help="SPSP setting")

    parser.add_argument("--yearly", default=False,
                        action='store_true',
                        help="yearly experiment")

    parser.add_argument("-c", "--client", default=False,
                        action='store_true',
                        help="run SPSP_CVaR client mode")

    parser.add_argument("--compact_report", default=False,
                        action="store_true",
                        help="SPSP_CVaR compact setting report")

    parser.add_argument("--general_report", default=False,
                        action="store_true",
                        help="SPSP_CVaR general setting report")

    args = parser.parse_args()
    if args.server:
        print("run SPSP_CVaR parameter server mode")
        print("setting:{}, yearly:{}".format(args.setting, args.yearly))
        parameter_server(args.setting, args.yearly)
    elif args.client:
        print("run SPSP_CVaR client mode")
        parameter_client()
    elif args.compact_report:
        print("SPSP CVaR compact setting report")
        aggregating_reports("compact")
    elif args.general_report:
        print("SPSP CVaR general setting report")
        aggregating_reports("general")
    else:
        raise ValueError("no mode is set.")
