# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""
import datetime as dt
from time import time
import numpy as np
import pandas as pd
import logging

from portfolio_programming.simulation.mixin import (PortfolioReportMixin,
                                                    ValidPortfolioParameterMixin)


class StagewisePortfolioSP(ValidPortfolioParameterMixin,
                           PortfolioReportMixin):
    def __init__(self, candidate_symbols,
                 max_portfolio_size,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 initial_risk_free_wealth,
                 buy_trans_fee=0.001425,
                 sell_trans_fee=0.004425,
                 start_date=dt.date(2005, 1, 3),
                 end_date=dt.date(2014, 12, 31),
                 rolling_horizon=200,
                 n_scenario=200,
                 bias_estimator=False,
                 report_path=r'.',
                 verbose=False):
        """
        stagewise portfolio stochastic programming

        Parameters:
        -------------
        candidate_symbols : list of symbols,
            The size of the candidate set is  n_stock.

        max_portfolio_size : positive integer
            The max number of stock we can invest in the portfolio.
            If the max_portfolio_size == n_stock, it degenerates to the
            linear programming.

        risk_rois : pandas.DataFrame, shape: (n_period, n_stock)
            The return of all stocks in the given intervals.
            The n_exp_period should be subset of the n_period.

        risk_free_rois : pandas.series, shape: (n_exp_period, )
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

        rolling_horizon : positive integer
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

        amount_pnl : pandas.Panel, shape: (3, n_exp_period, n_stock+1)
            Buying, selling and transaction fee amount of each asset in each
            period of the             simulation.

        gen_scenario_fail : pandas.Series, shape: (n_exp_period,)
            The scenario-generating function may fail in generating
            scenarios in some periods.
        """
        # valid number of symbols
        self.valid_dimension("n_stock", len(candidate_symbols),
                             risk_rois.shape[1])
        self.candidate_symbols = candidate_symbols
        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        # valid number of symbols
        self.valid_dimension("n_stock", len(candidate_symbols),
                             len(initial_risk_wealth))
        self.initial_risk_wealth = initial_risk_wealth
        self.initial_risk_free_wealth = initial_risk_free_wealth

        # valid transaction fee
        self.buy_trans_fee = buy_trans_fee
        self.valid_trans_fee(buy_trans_fee)
        self.valid_trans_fee(sell_trans_fee)
        self.sell_trans_fee = sell_trans_fee

        self.verbose = verbose

        # .loc() will contain the end_date element
        self.valid_trans_date(start_date, end_date)
        self.exp_risk_rois = risk_rois.loc[start_date:end_date]
        self.exp_risk_free_rois = risk_free_rois.loc[start_date:end_date]
        self.n_exp_period = self.exp_risk_rois.shape[0]
        self.exp_start_date = self.exp_risk_rois.index[0]
        self.exp_end_date = self.exp_risk_rois.index[self.n_exp_period - 1]

        self.n_stock = self.exp_risk_rois.shape[1]

        # date index in total data
        self.rolling_horizon = int(rolling_horizon)
        self.n_scenario = int(n_scenario)
        self.bias_estimator = bias_estimator
        self.start_date_idx = self.risk_rois.index.get_loc(
            self.exp_risk_rois.index[0])

        if self.start_date_idx < rolling_horizon:
            raise ValueError('There is no enough data for estimating '
                             'parameters.')

        # valid specific parameters added by users
        self.valid_specific_parameters()

        # results data
        # wealth DataFrame, shape: (n_exp_period, n_stock)
        self.wealth_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # buying,selling, and transaction_free amount panel,
        # shape: (buy or sell, n_exp_period, n_stock)
        self.amounts_pnl = pd.Panel(
            np.zeros((2, self.n_exp_period, self.n_stock)),
            index=("buy", "sell", "trans_fee"),
            major_axis=self.exp_risk_rois.index,
            minor_axis=self.exp_risk_rois.columns
        )

        # generatiing scenario error count, shape: (n_exp_period,)
        self.gen_scenario_fail = pd.Series(np.zeros(
            self.n_exp_period).astype(np.bool),
           index=self.exp_risk_rois.index)


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
        estimated_risk_rois: pandas.DataFrame, shape: (n_stock, n_scenario)
        """
        raise NotImplementedError('get_estimated_rois')

    def get_estimated_risk_free_rois(self, *arg, **kwargs):
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

    def get_trading_func_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        string
            Function name of the class
        """
        raise NotImplementedError('get_trading_func_name')

    def set_specific_period_action(self, *args, **kwargs):
        pass

    def add_results_to_reports(self, reports, *args, **kwargs):
        """ add Additional results to reports after a simulation """
        return reports

    def run(self):
        """
        run the simulation

        Returns:
        ----------------
        standard report
        """
        t0 = time()

        # get simulation name
        simulation_name = self.get_simulation_name()

        # initial wealth of each stock in the portfolio
        allocated_risk_wealth = self.initial_risk_wealth
        allocated_risk_free_wealth = self.initial_risk_free_wealth

        # count of generating scenario error
        estimated_risk_roi_error_count = 0

        for tdx in range(self.n_exp_period):
            t1 = time()
            # estimating next period rois, shape: (n_stock, n_scenario)
            try:
                estimated_risk_rois = self.get_estimated_risk_rois(
                    tdx=tdx,
                    trans_date=self.exp_risk_rois.index[tdx],
                    n_stock=self.n_stock,
                    window_length=self.rolling_horizon,
                    n_scenario=self.n_scenario,
                    bias=self.bias_estimator)

            except ValueError as e:
                print("generating scenario error: {}, {}".format(
                    self.exp_risk_rois.index[tdx], e))
                self.gen_scenario_fail[tdx] = True

            estimated_risk_free_rois = self.get_estimated_risk_free_rois(
                tdx=tdx,
                trans_date=self.exp_risk_rois.index[tdx],
                n_stock=self.n_stock,
                window_length=self.rolling_horizon,
                n_scenario=self.n_scenario,
                bias=self.bias_estimator)

            # generating scenarios success
            # using new scenarios in the SP
            if not self.gen_scenario_fail[tdx]:

                # determining the buy and sell amounts
                results = self.get_current_buy_sell_amounts(
                    tdx=tdx,
                    trans_date=self.exp_risk_rois.index[tdx],
                    estimated_risk_rois=estimated_risk_rois,
                    estimated_risk_free_roi=estimated_risk_free_rois,
                    allocated_risk_wealth=allocated_risk_wealth,
                    allocated_risk_free_wealth=allocated_risk_free_wealth
                )
                # record results
                self.set_specific_period_action(tdx=tdx, results=results)

                # buy and sell according results, shape: (n_stock, )
                buy_amounts = results["buy_amounts"]
                sell_amounts = results["sell_amounts"]

            # generating scenarios failed, and no action in this period
            else:
                # buy and sell nothing, shape:(n_stock, )
                buy_amounts = pd.Series(np.zeros(self.n_stock),
                                        index=self.candidate_symbols)
                sell_amounts = pd.Series(np.zeros(self.n_stock),
                                         index=self.candidate_symbols)
                estimated_risk_roi_error_count += 1

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
                (1 + self.exp_risk_rois.iloc[tdx]) *
                allocated_risk_wealth +
                self.buy_amounts_df.iloc[tdx] - self.sell_amounts_df.iloc[tdx]
            )
            self.risk_free_wealth.iloc[tdx] = (
                (1 + self.exp_risk_free_rois.iloc[tdx]) *
                allocated_risk_free_wealth -
                total_buy + total_sell
            )

            # update wealth
            allocated_risk_wealth = self.risk_wealth_df.iloc[tdx]
            allocated_risk_free_wealth = self.risk_free_wealth.iloc[tdx]

            print("[{}/{}] {} {} OK, scenario err cnt:{} "
                  "cur_wealth:{:.2f}, {:.3f} secs".format(
                tdx + 1, self.n_exp_period,
                self.exp_risk_rois.index[tdx].strftime("%Y%m%d"),
                func_name,
                estimated_risk_roi_error_count,
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
            self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[edx],
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
        reports['window_length'] = self.rolling_horizon
        reports['n_scenario'] = self.n_scenario
        reports['buy_amounts_df'] = self.buy_amounts_df
        reports['sell_amounts_df'] = self.sell_amounts_df
        reports['estimated_risk_roi_error'] = self.gen_scenario_fail
        reports['estimated_risk_roi_error_count'] = \
            self.gen_scenario_fail.sum()

        # add simulation time
        reports['simulation_time'] = time() - t0

        # user specified  additional elements to reports
        reports = self.add_results_to_reports(reports)

        print("{} OK n_stock:{}, [{}-{}], {:.4f}.secs".format(
            simulation_name, self.n_stock,
            self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[edx],
            time() - t0))

        return reports
