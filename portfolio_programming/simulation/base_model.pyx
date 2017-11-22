# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""
from datetime import (date, )
from time import time
import platform
import numpy as np
import pandas as pd



cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INTP_t


class SPTradingPortfolio(ValidPortfolioParameterMixin,
                         PortfolioReportMixin):
    def __init__(self, symbols,
                 risk_rois, risk_free_rois,
                 initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=0.001425,
                 double sell_trans_fee=0.004425,
                 start_date=date(2005, 1, 3),
                 end_date=date(2014, 12, 31),
                 int window_length=200,
                 int n_scenario=200, bias=False, verbose=False):
        """
        stepwise stochastic programming trading portfolio

        Parameters:
        -------------
        symbols: list of symbols, size: n_stock
        risk_rois: pandas.DataFrame, shape: (n_period, n_stock)
        risk_free_rois: pandas.series, shape: (n_exp_period, )
        initial_risk_wealth: pandas.series, shape: (n_stock,)
        initial_risk_free_wealth: float
        buy_trans_fee: float, 0<=value < 1,
            the fee will not change in the simulation
        sell_trans_fee: float, 0<=value < 1, the same as above
        start_date: datetime.date, first date of simulation
        end_date: datetime.date, last date of simulation
        window_length: integer, historical periods for estimated parameters
        n_scenario: integer, number of scenarios to generated
        bias: boolean, biased moment estimators or not
        verbose: boolean

        Data
        --------------
        risk_wealth_df: pandas.DataFrame, shape: (n_exp_period, n_stock)
        risk_free_wealth_df: pandas.Series, shape: (n_exp_period,)
        buy_amount_df: pandas.DataFrame, shape: (n_exp_period, n_stock)
        sell_amount_df: pandas.DataFrame, shape: (n_exp_period, n_stock)
        estimated_risk_roi_error: pandas.Series, shape: (n_exp_period,)
            - The scenarios generating function may be failed to generate
              scenarios in some periods, we record these periods if fails.
        trans_fee_loss: float, the cumulative loss of transaction fee.

        """
        # valid number of symbols
        self.valid_dimension("n_stock", len(symbols), risk_rois.shape[1])
        self.symbols = symbols
        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        # valid number of symbols
        self.valid_dimension("n_stock", len(symbols),
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
        self.window_length = int(window_length)
        self.n_scenario = int(n_scenario)
        self.bias_estimator = bias
        self.start_date_idx = self.risk_rois.index.get_loc(
            self.exp_risk_rois.index[0])

        if self.start_date_idx < window_length:
            raise ValueError('There is no enough data for estimating '
                             'parameters.')

        # valid specific parameters added by users
        self.valid_specific_parameters()

        # results data
        # risk wealth DataFrame, shape: (n_exp_period, n_stock)
        self.risk_wealth_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # risk_free Series, shape: (n_exp_period, )
        self.risk_free_wealth = pd.Series(np.zeros(self.n_exp_period),
                                          index=self.exp_risk_free_rois.index)

        # buying amount DataFrame, shape: (n_exp_period, n_stock)
        self.buy_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # selling amount DataFrame, shape: (n_exp_period, n_stock)
        self.sell_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # estimating error Series, shape: (n_period,)
        # The scenarios generating function may be failed to generate
        # scenarios in some periods, we record these periods if fails.
        self.estimated_risk_roi_error = pd.Series(np.zeros(
            self.n_exp_period).astype(np.bool),
            index=self.exp_risk_rois.index)

        # cumulative loss in transaction fee in the simulation
        self.trans_fee_loss = 0

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
        risk_free_roi: float
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

    def get_trading_func_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        func_name: str, Function name of the class
        """
        raise NotImplementedError('get_trading_func_name')

    def set_specific_period_action(self, *args, **kwargs):
        pass

    def add_results_to_reports(self, reports, *args, **kwargs):
        """ add Additional results to reports after a simulation """
        return reports

    def run(self):
        """
        run recourse programming simulation

        Returns:
        ----------------
        standard report
        """
        t0 = time()

        # get function name
        func_name = self.get_trading_func_name()

        # current wealth of each stock in the portfolio
        allocated_risk_wealth = self.initial_risk_wealth
        allocated_risk_free_wealth = self.initial_risk_free_wealth

        # count of generating scenario error
        estimated_risk_roi_error_count = 0

        for tdx in xrange(self.n_exp_period):
            t1 = time()
            # estimating next period rois, shape: (n_stock, n_scenario)
            try:
                estimated_risk_rois = self.get_estimated_risk_rois(
                    tdx=tdx,
                    trans_date=self.exp_risk_rois.index[tdx],
                    n_stock=self.n_stock,
                    window_length=self.window_length,
                    n_scenario=self.n_scenario,
                    bias=self.bias_estimator)

            except ValueError as e:
                print ("generating scenario error: {}, {}".format(
                    self.exp_risk_rois.index[tdx], e))
                self.estimated_risk_roi_error[tdx] = True

            estimated_risk_free_rois = self.get_estimated_risk_free_rois(
                tdx=tdx,
                trans_date=self.exp_risk_rois.index[tdx],
                n_stock=self.n_stock,
                window_length=self.window_length,
                n_scenario=self.n_scenario,
                bias=self.bias_estimator)

            # generating scenarios success
            if self.estimated_risk_roi_error[tdx] == False:

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

            # generating scenarios failed
            else:
                # buy and sell nothing, shape:(n_stock, )
                buy_amounts = pd.Series(np.zeros(self.n_stock),
                                        index=self.symbols)
                sell_amounts = pd.Series(np.zeros(self.n_stock),
                                         index=self.symbols)
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

            print ("[{}/{}] {} {} OK, scenario err cnt:{} "
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
            func_name,
            self.symbols,
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
        reports['window_length'] = self.window_length
        reports['n_scenario'] = self.n_scenario
        reports['buy_amounts_df'] = self.buy_amounts_df
        reports['sell_amounts_df'] = self.sell_amounts_df
        reports['estimated_risk_roi_error'] = self.estimated_risk_roi_error
        reports['estimated_risk_roi_error_count'] = \
            self.estimated_risk_roi_error.sum()

        # add simulation time
        reports['simulation_time'] = time() - t0

        # user specified  additional elements to reports
        reports = self.add_results_to_reports(reports)

        print ("{} OK n_stock:{}, [{}-{}], {:.4f}.secs".format(
            func_name, self.n_stock,
            self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[edx],
            time() - t0))

        return reports