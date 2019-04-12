# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chen1116@gmail.com>
"""

import datetime as dt
import glob
import logging
import multiprocessing as mp
import os
import pickle
import platform
import sys
import pandas as pd
from time import (time, sleep)

import numpy as np
import xarray as xr
import zmq

import portfolio_programming as pp
from portfolio_programming.simulation.run_spsp_cvar import run_SPSP_CVaR


def get_zmq_version():
    node = platform.node()
    print("Node:{} libzmq version is {}".format(node, zmq.zmq_version()))
    print("Node:{} pyzmq version is {}".format(node, zmq.__version__))


def _all_spsp_cvar_params(exp_name, setting, yearly=False):
    """
    "report_SPSP_CVaR_{}_{}_Mc{}_M{}_h{}_s{}_a{:.2f}_sdx{}_{}_{}.pkl".format(
        self.setting,
        self.group_name,
        self.n_symbol,
        self.max_portfolio_size,
        self.rolling_window_size,
        self.n_scenario,
        self.alpha,
        self.scenario_set_idx,
        self.exp_start_date.strftime("%Y%m%d"),
        self.exp_end_date.strftime("%Y%m%d"),
    )
    """
    REPORT_FORMAT = "report_SPSP_CVaR_{setting}_{group_name}_Mc{n_symbol}_M{max_portfolio_size}_h{rolling_window_size}_s{n_scenario}_a{alpha}_sdx{sdx}_{exp_start_date}_{exp_end_date}.pkl"

    if exp_name not in ('dissertation', 'stocksp_cor15'):
        raise ValueError('unknown exp_name:{}'.format(exp_name))

    if setting not in ('compact', 'general'):
        raise ValueError('Wrong setting: {}'.format(setting))

    #set_indices = (1, 2, 3)

    set_indices = (1,)

    if exp_name == "dissertation":
        if not yearly:
            # whole interval
            years = [(dt.date(2005, 1, 3), dt.date(2018, 12, 28))]
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
        max_portfolio_sizes = [5, ]
        group_symbols = pp.GROUP_SYMBOLS
        window_sizes = range(50, 240 + 10, 10)
        n_scenarios = [1000, ]
        alphas = ["{:.2f}".format(v / 100.) for v in range(50, 100, 5)]

        # dict comprehension
        # key: file_name, value: parameters
        if setting in ("compact", ):
            params = {
                REPORT_FORMAT.format(
                    setting=setting,
                    group_name=group_name,
                    n_symbol=len(symbols),
                    max_portfolio_size=m,
                    rolling_window_size=h,
                    n_scenario=s,
                    alpha=a,
                    sdx=sdx,
                    exp_start_date=s_date.strftime("%Y%m%d"),
                    exp_end_date=e_date.strftime("%Y%m%d")
                ): (exp_name, setting, group_name, m, h, s, float(a), sdx,
                    s_date, e_date)
                for group_name, symbols in group_symbols.items()
                for m in max_portfolio_sizes
                for h in window_sizes
                for s in n_scenarios
                for a in alphas
                for sdx in set_indices
                for s_date, e_date in years
            }
            return params

        elif setting == "general":
            return {
                REPORT_FORMAT.format(
                    setting=setting,
                    group_name=group_name,
                    n_symbol=len(symbols),
                    max_portfolio=m,
                    rolling_window_size=h,
                    n_scenario=s,
                    alpha=a,
                    sdx=sdx,
                    exp_start_date=s_date.strftime("%Y%m%d"),
                    exp_end_date=e_date.strftime("%Y%m%d"),
                ): (setting, group_name, len(symbols), m, h, s, a, sdx,
                    s_date, e_date)
                for group_name, symbols in group_symbols.items()
                for m in max_portfolio_sizes
                for h in window_sizes
                for s in n_scenarios
                for a in alphas
                for sdx in set_indices
                for s_date, e_date in years
            }


def checking_existed_spsp_cvar_report(exp_name, setting, yearly):
    """
    return unfinished experiment parameters.
    """
    if yearly:
        report_dir = os.path.join(pp.REPORT_DIR,
                                  "SPSP_CVaR_{}_yearly".format(setting))
    else:
        report_dir = pp.REPORT_DIR
        # report_dir = os.path.join(pp.REPORT_DIR,
        #                 "SPSP_CVaR_{}_20050103_20181228".format(setting))

    all_reports = _all_spsp_cvar_params(exp_name, setting, yearly)
    print("{} {} totally n_parameter: {}".format(
        exp_name, setting, len(all_reports)))

    os.chdir(report_dir)
    existed_reports = glob.glob("*.pkl")
    for report in existed_reports:
        all_reports.pop(report, None)

    # unfinished params
    return all_reports


def parameter_server(exp_name, setting, yearly):
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
     checking_existed_spsp_cvar_report(exp_name, setting, yearly).values()]
    progress_node_pid = set()
    progress_node_count = {}
    finished = {}
    print("Ready to serving, {} {} remaining {} n_parameter.".format(
        exp_name, setting, params.qsize()))

    svr_start_time = dt.datetime.now()
    t0 = time()

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
            progress_node_count[c_node]['req_time'] = dt.datetime.now()
        else:
            progress_node_count.setdefault(
                c_node, {"req_time": dt.datetime.now(), "cnt": 0})
            progress_node_count[c_node]['req_time'] = dt.datetime.now()
            progress_node_count[c_node]['cnt'] += 1

        # the progress set is not robust, because we don't track
        # if a process on a node is crashed or not.
        progress_node_pid.add(client_node_pid)

        print("server start time:{}, elapsed:{}\nremaining n_parameter:{}, "
              "".format(svr_start_time.strftime("%Y%m%d-%H:%M:%S"),
                        time() - t0, params.qsize()))

        print("progressing: {}".format(len(progress_node_pid)))
        for c_node, cnt in finished.items():
            print("node:{:<8} progress:{:>3} ,finish:{:>3} last req:{}".format(
                c_node, progress_node_count[c_node]['cnt'], cnt,
                progress_node_count[c_node]['req_time'].strftime(
                    "%Y%m%d-%H:%M:%S"))
            )

    print("end of serving, remaining {} parameters.".format(params.qsize()))
    socket.close()
    context.term()
    params.close()


def parameter_client(server_ip="140.117.168.49", max_reconnect_count=30):
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
    reconnect_count = 0
    while True:
        # send request to server
        socket.send_string(node_pid)

        # wait 10 seconds for server responding
        socks = dict(poll.poll(10000))

        if socks.get(socket) == zmq.POLLIN:
            # still connected
            reconnect_count = 0

            # receive parameters from server
            work = socket.recv_pyobj()
            print("{:<15} receiving: {}".format(
                str(dt.datetime.now()),
                work))
            run_SPSP_CVaR(*work)

        else:
            # no response from server, reconnected
            reconnect_count += 1
            sleep(10)
            if reconnect_count >= max_reconnect_count:
                break

            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
            poll.unregister(socket)

            # reconnection
            socket = context.socket(zmq.REQ)
            socket.connect(url)
            poll.register(socket, zmq.POLLIN)

            # socket.send_string(node_pid)
            print('{}, reconnect to {}'.format(dt.datetime.now(), url))

    socket.close()
    context.term()


def aggregating_reports(exp_name, setting, yearly=False):

    import arch.bootstrap.multiple_comparison as arch_comp

    if exp_name not in ('dissertation', 'stocksp_cor15'):
        raise ValueError('unknown exp_name:{}'.format(exp_name))

    if setting not in ("compact", "general"):
        raise ValueError("Unknown SPSP_CVaR setting: {}".format(setting))

    if exp_name == 'stocksp_cor15':
        if not yearly:
            # whole interval
            years = [[dt.date(2005, 1, 3), dt.date(2014, 12, 31)]]
            out_report_path = os.path.join(
                pp.DATA_DIR,
                "report_SPSP_CVaR_whole_{}_{}_{}_{}.nc".format(
                    exp_name, setting,
                    years[0][0].strftime("%Y%m%d"),
                    years[0][1].strftime("%Y%m%d")))

        else:
            years = [[dt.date(2005, 1, 3), dt.date(2005, 12, 30)],
                     [dt.date(2006, 1, 2), dt.date(2006, 12, 29)],
                     [dt.date(2007, 1, 2), dt.date(2007, 12, 31)],
                     [dt.date(2008, 1, 2), dt.date(2008, 12, 31)],
                     [dt.date(2009, 1, 5), dt.date(2009, 12, 31)],
                     [dt.date(2010, 1, 4), dt.date(2010, 12, 31)],
                     [dt.date(2011, 1, 3), dt.date(2011, 12, 30)],
                     [dt.date(2012, 1, 2), dt.date(2012, 12, 28)],
                     [dt.date(2013, 1, 2), dt.date(2013, 12, 31)],
                     [dt.date(2014, 1, 2), dt.date(2014, 12, 31)]
            ]
            out_report_path = os.path.join(pp.DATA_DIR,
                                           "report_SPSP_CVaR_yearly_{}_{}_{}_{"
                                           "}.nc".format(
                                               exp_name,
                                               setting,
                                               years[0][0].strftime("%Y%m%d"),
                                               years[-1][1].strftime("%Y%m%d")))

        intervals = ["{}_{}".format(s.strftime("%Y%m%d"), e.strftime("%Y%m%d"))
                     for s, e in years]
        set_indices = [1, 2, 3]
        group_names = []
        max_portfolio_sizes = range(5, 50 + 5, 5)
        window_sizes = range(60, 240 + 10, 10)
        n_scenarios = [200, ]
        alphas = ["{:.2f}".format(v / 100.) for v in range(50, 100, 5)]

    elif exp_name == 'dissertation':
        if not yearly:
            # whole interval
            years = [[dt.date(2005, 1, 3), dt.date(2018, 12, 28)]]

            out_report_path = os.path.join(
                pp.DATA_DIR,
                "report_SPSP_CVaR_whole_{}_{}_{}_{}.nc".format(
                    exp_name, setting,
                    years[0][0].strftime("%Y%m%d"),
                    years[0][1].strftime("%Y%m%d")))
        else:
            # yearly interval
            years = [[dt.date(2005, 1, 3), dt.date(2005, 12, 30)],
                     [dt.date(2006, 1, 2), dt.date(2006, 12, 29)],
                     [dt.date(2007, 1, 2), dt.date(2007, 12, 31)],
                     [dt.date(2008, 1, 2), dt.date(2008, 12, 31)],
                     [dt.date(2009, 1, 5), dt.date(2009, 12, 31)],
                     [dt.date(2010, 1, 4), dt.date(2010, 12, 31)],
                     [dt.date(2011, 1, 3), dt.date(2011, 12, 30)],
                     [dt.date(2012, 1, 2), dt.date(2012, 12, 28)],
                     [dt.date(2013, 1, 2), dt.date(2013, 12, 31)],
                     [dt.date(2014, 1, 2), dt.date(2014, 12, 31)],
                     [dt.date(2015, 1, 5), dt.date(2015, 12, 31)],
                     [dt.date(2016, 1, 4), dt.date(2016, 12, 30)],
                     [dt.date(2017, 1, 3), dt.date(2017, 12, 29)]
                     ]
            out_report_path = os.path.join(pp.DATA_DIR,
                                       "report_SPSP_CVaR_yearly_{}_{}_{}_{"
                                       "}.nc".format(
                                           exp_name,
                                           setting,
                                           years[0][0].strftime("%Y%m%d"),
                                           years[-1][1].strftime("%Y%m%d")))

        intervals = ["{}_{}".format(s.strftime("%Y%m%d"), e.strftime("%Y%m%d"))
                     for s, e in years]
        set_indices = [1, ]
        group_names = list(pp.GROUP_SYMBOLS.keys())
        max_portfolio_sizes = [5, ]
        window_sizes = range(50, 240 + 10, 10)
        n_scenarios = [1000, ]
        alphas = ["{:.2f}".format(v / 100.) for v in range(50, 100, 5)]

    originals = [
        'initial_wealth', 'final_wealth',
        'cum_roi', 'daily_roi', 'daily_mean_roi',
        'daily_std_roi', 'daily_skew_roi', 'daily_ex-kurt_roi',
        'Sharpe', 'Sortino_full', 'Sortino_partial'
    ]
    # additional attributes
    additionals = ['annual_roi', 'daily_VSS', 'SPA_c']
    attributes = originals + additionals

    report_xarr = xr.DataArray(
        np.zeros((len(years),
                  len(group_names),
                  len(set_indices),
                  len(max_portfolio_sizes),
                  len(window_sizes),
                  len(alphas),
                  len(attributes)
                  )),
        dims=("interval",
              "group_name",
              "scenario_set_idx",
              "max_portfolio_size",
              "rolling_window_size",
              "alpha",
              "attribute"),
        coords=(intervals,
                group_names,
                set_indices,
                max_portfolio_sizes,
                window_sizes,
                alphas,
                attributes)
    )
    t0 = time()
    # key: report_name, value: parameters
    report_dict = _all_spsp_cvar_params(exp_name, setting, yearly)
    report_count = 0
    no_report_count_params = []
    no_report_count = 0
    parent_dir = pp.REPORT_DIR

    for idx, (name, param) in enumerate(report_dict.items()):
        t1 = time()
        path = os.path.join(parent_dir, name)
        print(path)
        exp_name, setting, grp, m, h, s, a, sdx, s_date, e_date = param
        interval = "{}_{}".format(s_date.strftime("%Y%m%d"),
                                  e_date.strftime("%Y%m%d"))
        alpha = "{:.2f}".format(a)
        try:
            report = pd.read_pickle(path)

            for attr in originals:
                report_xarr.loc[
                    interval, grp, sdx, m, h, alpha, attr] = report[attr]

            year_count = (e_date.year - s_date.year) + 1
            for attr in additionals:
                if attr == 'annual_roi':
                    val = np.power(report['cum_roi'] + 1, 1. / year_count) - 1
                elif attr == 'daily_VSS':
                    risks = report['estimated_risk_xarr']
                    val = float(risks.loc[:, 'VSS'].mean() / report[
                        'initial_wealth'])
                elif attr == 'SPA_c':
                    dec_xarr = report['decision_xarr']
                    wealth_arr = dec_xarr.loc[:, :, 'wealth'].sum(
                        axis=1).to_series()
                    rois = wealth_arr.pct_change()
                    rois[0] = 0
                    # print(rois.values)
                    spa_value = 0
                    for _ in range(3):
                        spa = arch_comp.SPA(rois.values,
                                            np.zeros(rois.size),
                                            reps=1000)
                        spa.seed(np.random.randint(0, 2 ** 31 - 1))
                        spa.compute()
                        # preserve the worse p_value
                        if spa.pvalues[1] > spa_value:
                            spa_value = spa.pvalues[1]
                    val = spa_value

                report_xarr.loc[
                    interval, grp, sdx, m, h, alpha, attr] = val

            report_count += 1
            print("[{}/{}] {} {:.2%} elapsed:{:.2f}/{:.2f} secs".format(
                idx + 1, len(report_dict),
                report['simulation_name'],
                report['cum_roi'],
                time() - t1,
                time() - t0
            ))

        except FileNotFoundError:
            no_report_count_params.append(name)
            no_report_count += 1
            continue
        except Exception as e:
            print("{} Error: {}".format(name, e))
            sys.exit(-1)

    for rp in no_report_count_params:
        print("no data:", rp)

    print("report count:{}, no report count:{}".format(
        report_count, no_report_count))

    report_xarr.to_netcdf(out_report_path)


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

    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        default="dissertation",
        choices=["dissertation", "stocksp_cor15"],
        help="name of the experiment",
    )

    parser.add_argument("--setting", type=str,
                        choices=("compact", "general", "compact_mu0"),
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

    parser.add_argument("--unfinished_param", default=False,
                        action="store_true")

    args = parser.parse_args()
    if args.server:
        print("run SPSP_CVaR parameter server mode")
        print("exp_name: {}, setting:{}, yearly:{}".format(
            args.exp_name, args.setting, args.yearly))
        parameter_server(args.exp_name, args.setting, args.yearly)
    elif args.client:
        print("run SPSP_CVaR client mode")
        parameter_client()
    elif args.compact_report:
        print("SPSP CVaR compact setting report")
        aggregating_reports(args.exp_name, "compact", args.yearly)
    elif args.general_report:
        print("SPSP CVaR general setting report")
        aggregating_reports(args.exp_name, "general", args.yearly)
    elif args.unfinished_param:
        params_dict = checking_existed_spsp_cvar_report(args.setting,
                                                       args.yearly)
        for rp in params_dict.keys():
            print("no data:", rp)
        print("no data count: {}.".format(len(params_dict)))
    else:
        raise ValueError("no mode is set.")
