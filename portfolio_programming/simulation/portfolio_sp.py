# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import datetime as dt
import time
import numpy as np
import pandas as pd
import logging

from portfolio_programming.simulation.mixin import (
    PortfolioSPReportMixin, ValidMixin)

import portfolio_programming as pp


class BaseStagewisePortfolioSP(ValidMixin,
                               PortfolioSPReportMixin):
    def __init__(self,
                 candidate_symbols,
                 max_portfolio_size,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 initial_risk_free_wealth,
                 buy_trans_fee=0.001425,
                 sell_trans_fee=0.004425,
                 sim_start_date=pp.SIM_START_DATE,
                 sim_end_date=pp.SIM_END_DATE,
                 rolling_window_size=200,
                 n_scenario=200,
                 bias_estimator=False,
                 report_path=r'.',
                 verbose=False):
        """
        stagewise portfolio stochastic programming base model

        Parameters:
        -------------
        candidate_symbols : list of symbols,
            The size of the candidate set is n_stock.

        max_portfolio_size : positive integer
            The max number of stock we can invest in the portfolio.
            The model is the mixed integer linear programming, however,
            if the max_portfolio_size == n_stock, it degenerates to the
            linear programming.

        risk_rois : pandas.DataFrame, shape: (n_period, n_stock)
            The return of all stocks in the given intervals.
            The n_exp_period should be subset of the n_period.

        risk_free_rois : pandas.series or numpy.array, shape: (n_exp_period, )
            The return of risk-free asset, usually all zeros.

        initial_risk_wealth : pandas.series, shape: (n_stock,)
            The invested wealth of the stocks in the candidate set.

        initial_risk_free_wealth : float
            The initial wealth in the bank or the risky-free asset.

        buy_trans_fee : float
            The fee usually not change in the simulation.

        sell_trans_fee : float,
             The fee usually not change in the simulation.

        start_date : datetime.date
            The first trading date (not the calendar day) of simulation.

        end_date : datetime.date
             The last trading date (not the calendar day) of simulation.

        rolling_window_size : positive integer
            The historical trading days for estimating statistics.

        n_scenario : positive integer
            The number of scenarios to generate.

        bias_estimator : boolean
            Using biased moment estimators or not.

        report_path : string
            The performance report file path of the simulation.

        verbose : boolean

        Data
        --------------
        wealth_df : pandas.DataFrame, shape: (n_exp_period, n_stock+1)
            The risky and risk-free assets wealth in each period of the
            simulation.
            The risk-free asset symbol is self.risk_free_symbol

        amount_pnl : pandas.Panel, shape: (n_exp_period. n_stock+1, 3)
            Buying, selling and transaction fee amount of each asset in each
            period of the simulation.


        estimated_risks: pandas.DataFrame, shape: (n_exp_period, 3)
            The estimated CVaR, VaR and number of gen_scenario_fail in
            the simulation.
            The scenario-generating function may fail in generating
            scenarios in some periods.

        """

        self.risk_free_symbol = 'risk_free'

        self.valid_dimension("n_stock", len(candidate_symbols),
                             risk_rois.shape[1])

        self.n_stock = len(candidate_symbols)
        self.candidate_symbols = candidate_symbols
        self.periods = risk_rois.index

        self.valid_nonnegative_value("max_portfolio_size", max_portfolio_size)
        self.max_portfolio_size = max_portfolio_size

        if max_portfolio_size > self.n_stock:
            raise ValueError("The portfolio size {} can't large than the "
                             "size of candidate set. {}.".format(
                max_portfolio_size, self.n_stock))

        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        self.valid_dimension("n_stock", len(candidate_symbols),
                             len(initial_risk_wealth))

        self.valid_nonnegative_list(
            ("initial_risk_wealth", initial_risk_free_wealth))
        self.initial_risk_wealth = initial_risk_wealth

        self.valid_nonnegative_value("initial_risk_free_wealth",
                                     initial_risk_free_wealth)
        self.initial_risk_free_wealth = initial_risk_free_wealth

        self.valid_range_value("buy_trans_fee", buy_trans_fee, 0, 1)
        self.buy_trans_fee = buy_trans_fee

        self.valid_range_value("sell_trans_fee", sell_trans_fee, 0, 1)
        self.sell_trans_fee = sell_trans_fee

        self.verbose = verbose

        # .loc() will contain the end_date element
        self.valid_trans_date(sim_start_date, sim_end_date)

        self.sim_risk_rois = risk_rois.loc[sim_start_date:sim_end_date]
        self.sim_risk_free_rois = risk_free_rois.loc[
                                  sim_start_date:sim_end_date]
        self.n_exp_period = self.sim_risk_rois.shape[0]
        self.sim_start_date = self.sim_risk_rois.index[0]
        self.sim_end_date = self.sim_risk_rois.index[self.n_exp_period - 1]

        self.n_stock = self.sim_risk_rois.shape[1]

        # date index in total data
        self.rolling_window_size = int(rolling_window_size)
        self.n_scenario = int(n_scenario)
        self.bias_estimator = bias_estimator
        self.report_path = report_path

        self.sim_start_date_idx = self.risk_rois.index.get_loc(
            self.sim_risk_rois.index[0])

        if self.sim_start_date_idx < rolling_window_size:
            raise ValueError('There is no enough data for estimating '
                             'parameters.')

        # valid specific parameters added by users
        self.valid_specific_parameters()

        # results data
        # wealth DataFrame, shape: (n_exp_period, n_stock+1)
        self.wealth_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.sim_risk_rois.index,
            columns=candidate_symbols + [self.risk_free_symbol, ]
        )

        # buying,selling, and transaction_free amount panel,
        # shape: (n_exp_period, n_stock+1, 3)
        self.amounts_pnl = pd.Panel(
            np.zeros((self.n_exp_period, self.n_stock + 1, 3)),
            index=self.sim_risk_rois.index,
            major_axis=self.candidate_symbols + [self.risk_free_symbol, ],
            minor_axis=("buy", "sell", "trans_fee"),
        )

        # estimated CVaR, VARs, and gen_scenario_fail
        self.estimated_risks = pd.DataFrame(
            np.zeros(self.n_exp_period, 3),
            index=self.sim_risk_rois.index,
            columns=('CVaR', 'VaR', 'gen_scenario_fail'))

    def valid_specific_parameters(self, *args, **kwargs):
        """
        implemented by user
        the function will be called in the __init__
        """
        pass

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        estimating next period risky assets rois,
        implemented by user

        Returns:
        ----------------------------
        pandas.DataFrame, shape: (n_stock, n_scenario)
        """
        raise NotImplementedError('get_estimated_rois')

    def get_estimated_risk_free_roi(self, *arg, **kwargs):
        """
        estimating next period risk free asset rois,
        implemented by user, and it should return a float number.

        Returns:
        ------------------------------
        risk_free_roi : float
        """
        raise NotImplementedError('get_estimated_risk_free_roi')

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """
        stochastic programming for determining current period
        buy amounts and sell amounts by using historical data.

        implemented by user, and it must return a dict contains
        at least the following two elements:
        {
            "buy_amounts": buy_amounts, pandas.Series, shape: (n_stock, )
            "sell_amounts": sell_amounts, , pandas.Series, shape: (n_stock, )
        }
        """
        raise NotImplementedError('get_current_buy_sell_amounts')

    def get_simulation_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        string
           simulation name of this experiment
        """
        raise NotImplementedError('get_simulation_name')

    def set_specific_action(self, *args, **kwargs):
        """ Set specific action in each period. """
        pass

    def add_to_reports(self, reports, *args, **kwargs):
        """ add Additional results to reports after a simulation """
        return reports

    def run(self):
        """
        run the simulation

        Returns:
        ----------------
        standard report
        """
        t0 = time.time()

        # get simulation name
        simulation_name = self.get_simulation_name()

        # initial wealth of each stock in the portfolio
        allocated_risk_wealth = self.initial_risk_wealth
        allocated_risk_free_wealth = self.initial_risk_free_wealth

        gen_scenario_cnt = 0
        for tdx in range(self.n_exp_period):
            t1 = time.time()
            curr_date = self.sim_risk_rois.index[tdx]

            # flag of generating scenario error
            gen_scenario_failed = False

            # estimating next period risky rois, shape: (n_stock, n_scenario)
            try:
                # the scenario generation may be failed
                estimated_risk_rois = self.get_estimated_risk_rois(
                    tdx=tdx,
                    trans_date=curr_date,
                    n_stock=self.n_stock,
                    rolling_horizon=self.rolling_window_size,
                    n_scenario=self.n_scenario,
                    bias_estimator=self.bias_estimator)

            except ValueError as e:
                logging.warning("generating scenario error: {}, {}".format(
                    curr_date, e))
                gen_scenario_failed = True
                gen_scenario_cnt += 1

            # estimating next period risk_free roi, return float
            estimated_risk_free_roi = self.get_estimated_risk_free_roi(
                tdx=tdx,
                trans_date=curr_date,
                n_stock=self.n_stock,
                window_length=self.rolling_window_size,
                n_scenario=self.n_scenario,
                bias=self.bias_estimator)

            # generating risky_roi scenarios success
            # using new scenarios in the SP
            if not gen_scenario_failed:
                # determining the buy and sell amounts
                pg_results = self.get_current_buy_sell_amounts(
                    tdx=tdx,
                    trans_date=curr_date,
                    estimated_risk_rois=estimated_risk_rois,
                    estimated_risk_free_roi=estimated_risk_free_roi,
                    allocated_risk_wealth=allocated_risk_wealth,
                    allocated_risk_free_wealth=allocated_risk_free_wealth
                )
                # record results
                # self.set_specific_action(tdx=tdx, results=results)

                # # buy and sell according results, shape: (n_stock, )
                buy_amounts = pg_results["buy_amounts"]
                sell_amounts = pg_results["sell_amounts"]
                estimated_var = pg_results["estimated_var"]
                estimated_cvar = pg_results["estimated_var"]

            # generating scenarios failed, and no action in this period
            else:
                self.estimated_risks.loc[curr_date, 'gen_scenario_fail'] = 1

                # buy and sell nothing, shape:(n_stock, )
                buy_amounts = pd.Series(np.zeros(self.n_stock),
                                        index=self.candidate_symbols)
                sell_amounts = pd.Series(np.zeros(self.n_stock),
                                         index=self.candidate_symbols)
                estimated_var = 0
                estimated_cvar = 0

            # record buy and sell amounts
            self.buy_amounts_df.iloc[tdx] = buy_amounts
            self.sell_amounts_df.iloc[tdx] = sell_amounts

            # record the transaction loss
            buy_amounts_sum = buy_amounts.sum()
            sell_amounts_sum = sell_amounts.sum()
            self.trans_fee_loss += (
                    buy_amounts_sum * self.buy_trans_fee +
                    sell_amounts_sum * self.sell_trans_fee
            )

            # buy and sell amounts consider the transaction cost
            total_buy = (buy_amounts_sum * (1 + self.buy_trans_fee))
            total_sell = (sell_amounts_sum * (1 - self.sell_trans_fee))

            # capital allocation
            self.risk_wealth_df.iloc[tdx] = (
                    (1 + self.sim_risk_rois.iloc[tdx]) *
                    allocated_risk_wealth +
                    self.buy_amounts_df.iloc[tdx] - self.sell_amounts_df.iloc[
                        tdx]
            )
            self.risk_free_wealth.iloc[tdx] = (
                    (1 + self.sim_risk_free_rois.iloc[tdx]) *
                    allocated_risk_free_wealth -
                    total_buy + total_sell
            )

            # update wealth
            allocated_risk_wealth = self.risk_wealth_df.iloc[tdx]
            allocated_risk_free_wealth = self.risk_free_wealth.iloc[tdx]

            logging.info("[{}/{}] {} {} OK, scenario err cnt:{} "
                         "cur_wealth:{:.2f}, {:.3f} secs".format(
                tdx + 1, self.n_exp_period,
                self.sim_risk_rois.index[tdx].strftime("%Y%m%d"),
                simulation_name,
                gen_scenario_error_cnt,
                (self.risk_wealth_df.iloc[tdx].sum() +
                 self.risk_free_wealth.iloc[tdx]),
                time() - t1))

        # end of iterations, computing statistics
        edx = self.n_exp_period - 1
        final_wealth = (self.risk_wealth_df.iloc[edx].sum() +
                        self.risk_free_wealth[edx])

        # get reports
        reports = self.get_performance_report(
            simulation_name,
            self.candidate_symbols,
            self.sim_risk_rois.index[0],
            self.sim_risk_rois.index[edx],
            self.buy_trans_fee,
            self.sell_trans_fee,
            (self.initial_risk_wealth.sum() + self.initial_risk_free_wealth),
            final_wealth,
            self.n_exp_period,
            self.trans_fee_loss,
            self.risk_wealth_df,
            self.risk_free_wealth,
        )

        # model additional elements to reports
        reports['window_length'] = self.rolling_window_size
        reports['n_scenario'] = self.n_scenario
        reports['buy_amounts_df'] = self.buy_amounts_df
        reports['sell_amounts_df'] = self.sell_amounts_df
        reports['estimated_risk_roi_error'] = self.gen_scenario_fail
        reports['estimated_risk_roi_error_count'] = \
            self.gen_scenario_fail.sum()

        # add simulation time
        reports['simulation_time'] = time.time() - t0

        # user specified  additional elements to reports
        reports = self.add_to_reports(reports)

        logging.info("{} OK n_stock:{}, [{}-{}], {:.4f}.secs".format(
            simulation_name, self.n_stock,
            self.sim_risk_rois.index[0],
            self.sim_risk_rois.index[edx],
            time.time() - t0))

        return reports
