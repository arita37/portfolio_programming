# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import logging
import sys
import platform
import numpy as np
import xarray as xr
import zmq
import os
import glob
import multiprocess as mp
import datetime as dt
from time import (time, sleep)

import portfolio_programming as pp
import portfolio_programming.simulation.spsp_cvar
from portfolio_programming.simulation.spsp_cvar import (
    NER_SPSP_CVaR, NIR_SPSP_CVaR)


def get_zmq_version():
    node = platform.node()
    print("Node:{} libzmq version is {}".format(node, zmq.zmq_version()))
    print("Node:{} pyzmq version is {}".format(node, zmq.__version__))


def all_nr_spsp_cvar_params(exp_name, regret_type):
    """
    "report_NR_SPSP_CVaR_{}_{:.2f}_{}_{}_s{}_sdx{}_{}_{}.pkl".format(
        self.nr_strategy,
        self.nr_strategy_param,
        self.group_name,
        self.expert_group_name,
        self.n_scenario,
        self.scenario_set_idx,
        self.exp_start_date.strftime("%Y%m%d"),
        self.exp_end_date.strftime("%Y%m%d"),
    )
    """
    if regret_type == 'external':
        REPORT_FORMAT = "report_NR_SPSP_CVaR_{nr_strategy}_{nr_strategy_param:.2f}_{group_name}_{expert_group_name}_s{n_scenario}_sdx{scenario_set_idx}_{exp_start_date}_{exp_end_date}.pkl"

        strategy_params = [[s, p] for s in ('EG', 'EXP')
                           for p in (0.01, 0.1, 1)]
        strategy_params.extend([[s, p] for s in ('POLY',) for p in (2, 3)])

    elif regret_type == 'internal':
        REPORT_FORMAT = "report_NIR_SPSP_CVaR_{nr_strategy}_{nr_strategy_param:.2f}_{group_name}_{expert_group_name}_s{n_scenario}_sdx{scenario_set_idx}_{exp_start_date}_{exp_end_date}.pkl"
        strategy_params = [[s, p] for s in ('EXP',)
                           for p in (0.01, 0.1, 1)]
        strategy_params.extend([[s, p] for s in ('POLY',) for p in (2, 3)])
    else:
        raise ValueError('unknown regret type:', regret_type)

    if exp_name not in ('dissertation',):
        raise ValueError('unknown exp_name:{}'.format(exp_name))

    group_params = {
        'TWG1': 'h140-200-10_a85-95-5',
        'TWG2': 'h190-240-10_a55-75-5',
        'TWG3': 'h60-100-10_a75-90-',
        'TWG4': 'h100-140-10_a55-75-5',
        'TWG5': 'h60-90-10_a50-75-5',
        'TWG6': 'h200-240-10_a50-70-5',
        'USG1': 'h200-240-10_a50-65-5',
        'USG2': 'h170-240-10_a50-70-5',
        'USG3': 'h170-220-10_a80-95-5',
        'USG4': 'h60-90-10_a75-90-5',
        'USG5': 'h80-130-10_a75-90-5',
        'USG6': 'h180-240-10_a50-70-5'
    }

    set_indices = (1, )
    n_scenarios = (1000,)
    if exp_name == "dissertation":
        years = [(dt.date(2005, 1, 3), dt.date(2018, 12, 28))]

    # parameters
    params = {
        REPORT_FORMAT.format(
            nr_strategy=s,
            nr_strategy_param=p,
            group_name=group_name,
            expert_group_name=exp_group_name,
            n_scenario=n_scenario,
            scenario_set_idx=sdx,
            exp_start_date=s_date,
            exp_end_date=e_date
        ): (s, p, group_name, exp_group_name, n_scenario, sdx, s_date, e_date)
        for s, p in strategy_params
        for group_name, exp_group_name in group_params
        for n_scenario in n_scenarios
        for sdx in set_indices
        for s_date, e_date in years
    }
    return params


def checking_existed_spsp_cvar_report(exp_name, regret_type):
    all_reports = all_nr_spsp_cvar_params(exp_name, regret_type)

    report_dir = pp.NRSPSPCVaR_DIR
    print("{} {} totally n_parameter: {}".format(
        exp_name, regret_type, len(all_reports)))

    os.chdir(report_dir)
    existed_reports = glob.glob("*.pkl")
    for report in existed_reports:
        all_reports.pop(report, None)

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
            run_NR_SPSP_CVaR(*work)

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


def get_experts(expert_group_name):
    """
    pattern 'h100-150-10_a50-70-5
    """
    h, a = expert_group_name.split('_')
    h_params = h[1:].split('-')
    a_params = a[1:].split('-')
    h_start, h_end, h_inc = map(int, h_params)
    a_start, a_end, a_inc = map(int, a_params)
    experts = [(h, a / 100)
               for h in range(h_start, h_end + h_inc, h_inc)
               for a in range(a_start, a_end + a_inc, a_inc)]
    return experts


def run_NR_SPSP_CVaR(exp_name, regret_type,
                     nr_strategy, nr_param, expert_group_name,
                     group_name, n_scenario, scenario_set_idx,
                     exp_start_date, exp_end_date):
    market = group_name[:2]
    if market == "TW":
        risky_roi_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)
    elif market == "US":
        risky_roi_xarr = xr.open_dataarray(pp.DJIA_2005_NC)

    candidate_symbols = pp.GROUP_SYMBOLS[group_name]
    n_symbol = len(candidate_symbols)
    risky_rois = risky_roi_xarr.loc[exp_start_date:exp_end_date,
                 candidate_symbols, 'simple_roi']
    exp_trans_dates = risky_rois.get_index('trans_date')
    # print('exp_trans_dates:', exp_trans_dates)
    n_exp_dates = len(exp_trans_dates)
    risk_free_rois = xr.DataArray(np.zeros(n_exp_dates),
                                  coords=(exp_trans_dates,))
    initial_risk_wealth = xr.DataArray(np.zeros(n_symbol),
                                       dims=('symbol',),
                                       coords=(candidate_symbols,))
    initial_risk_free_wealth = 100
    print(exp_name, nr_strategy, nr_param, expert_group_name,
          group_name, n_scenario, scenario_set_idx,
          exp_start_date, exp_end_date)

    experts = get_experts(expert_group_name)

    if regret_type == "external":
        obj = NER_SPSP_CVaR
    elif regret_type == 'internal':
        obj = NIR_SPSP_CVaR

    instance = obj(
        nr_strategy,
        nr_param,
        expert_group_name,
        experts,
        group_name,
        candidate_symbols,
        risky_rois,
        risk_free_rois,
        initial_risk_wealth,
        initial_risk_free_wealth,
        start_date=exp_trans_dates[0],
        end_date=exp_trans_dates[-1],
        n_scenario=n_scenario,
        scenario_set_idx=scenario_set_idx,
        print_interval=1
    )
    instance.run()


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(filename)15s %(levelname)10s %(asctime)s\n'
               '%(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        level=logging.INFO)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--regret", type=str, help="regret type")
    parser.add_argument("--nr_strategy", type=str, help="no-regret strategy")
    parser.add_argument("--nr_param", type=float,
                        help="no-regret strategy parameter")
    parser.add_argument("--expert_group_name", type=str)
    parser.add_argument("-g", "--group_name", type=str)
    parser.add_argument("-s", "--n_scenario", type=int, choices=[200, 1000],
                        default=1000, help="number of scenario")
    parser.add_argument("--sdx", type=int, choices=range(1, 4), default=1,
                        help="pre-generated scenario set index.")

    args = parser.parse_args()
    run_NR_SPSP_CVaR('dissertation', args.regret, args.nr_strategy,
                     args.nr_param, args.expert_group_name, args.group_name,
                     args.n_scenario, args.sdx, '20050103', '20181228')
