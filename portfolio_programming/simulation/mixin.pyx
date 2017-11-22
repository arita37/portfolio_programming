# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""


class PortfolioReportMixin(object):
    @staticmethod
    def get_performance_report(
            simulation_name, trading_func_name, symbols, start_date,
                               end_date,
                               buy_trans_fee, sell_trans_fee,
                               initial_wealth, final_wealth, n_exp_period,
                               trans_fee_loss, wealth_df):
        """
        standard reports

        Parameters:
        ------------------
        simulation_name : string
        trading_func_name : string
        symbols: list of string
            the candidate symbols in the simulation
        start_date, end_date: datetime.date
            the starting and ending days of the simulation
        buy_trans_fee, sell_trans_fee: float
            the transaction fee in the simulation
        initial_wealth, final_wealth: float
        n_exp_period: integer
        trans_fee_loss: float
        wealth_df: pandas.DataFrame, shape:(n_exp_period, n_stock + 1)
            the wealth series of each symbols in the simulation.
            It includes the risky and risk-free asset.
        """
        reports = {}
        # platform information
        reports['os_uname'] = "|".join(platform.uname())

        # return analysis
        reports['func_name'] = func_name
        reports['symbols'] = symbols
        reports['start_date'] = start_date
        reports['end_date'] = end_date
        reports['buy_trans_fee'] = buy_trans_fee
        reports['sell_trans_fee'] = sell_trans_fee
        reports['initial_wealth'] = initial_wealth
        reports['final_wealth'] = final_wealth
        reports['n_exp_period'] = n_exp_period
        reports['trans_fee_loss'] = trans_fee_loss
        reports['wealth_df'] = risk_wealth_df
        reports['risk_free_wealth'] = risk_free_wealth_arr

        # analysis
        reports['n_stock'] = len(symbols)
        reports['cum_roi'] = final_wealth / initial_wealth - 1.
        reports['daily_roi'] = np.power(final_wealth / initial_wealth,
                                        1. / n_exp_period) - 1

        # wealth_arr, shape: (n_stock,)
        wealth_arr = risk_wealth_df.sum(axis=1) + risk_free_wealth_arr
        wealth_daily_rois = wealth_arr.pct_change()
        wealth_daily_rois[0] = 0

        reports['daily_mean_roi'] = wealth_daily_rois.mean()
        reports['daily_std_roi'] = wealth_daily_rois.std()
        reports['daily_skew_roi'] = wealth_daily_rois.skew()

        # excess Kurtosis
        reports['daily_kurt_roi'] = wealth_daily_rois.kurt()
        reports['sharpe'] = sharpe(wealth_daily_rois)
        reports['sortino_full'], reports['sortino_full_semi_std'] = \
            sortino_full(wealth_daily_rois)

        reports['sortino_partial'], reports['sortino_partial_semi_std'] = \
            sortino_partial(wealth_daily_rois)

        reports['max_abs_drawdown'] = maximum_drawdown(wealth_arr)

        # statistics test
        # SPA test, benchmark is no action
        spa = SPA(wealth_daily_rois, np.zeros(wealth_arr.size), reps=1000)
        spa.seed(np.random.randint(0, 2 ** 31 - 1))
        spa.compute()
        reports['SPA_l_pvalue'] = spa.pvalues[0]
        reports['SPA_c_pvalue'] = spa.pvalues[1]
        reports['SPA_u_pvalue'] = spa.pvalues[2]

        return reports


class ValidPortfolioParameterMixin(object):
    @staticmethod
    def valid_trans_fee(trans_fee):
        """
        Parameter:
        -------------
        trans_fee: float
        """
        if not 0 <= trans_fee <= 1:
            raise ValueError("wrong trans_fee: {}".format(trans_fee))

    @staticmethod
    def valid_dimension(str dim1_name, int dim1, int dim2):
        """
        Parameters:
        -------------
        dim1, dim2: positive integer
        dim1_name, str
        """
        if dim1 != dim2:
            raise ValueError("mismatch {} dimension: {}, {}".format(
                dim1_name, dim1, dim2))

    @staticmethod
    def valid_trans_date(start_date, end_date):
        """
        Parameters:
        --------------
        start_date, end_date: datetime.date
        """
        if start_date > end_date:
            raise ValueError("wrong transaction interval, start:{}, "
                             "end:{})".format(start_date, end_date))
