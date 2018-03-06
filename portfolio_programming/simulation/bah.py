# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

from time import time

from spsp_cvar import ValidMixin


class BAHPortfolio(ValidMixin):

    def __init__(self,
                 symbols,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 initial_risk_free_wealth,
                 buy_trans_fee=0.001425,
                 sell_trans_fee=0.004425,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 ):
        """
        uniform buy-and-hold portfolio
        that is, one invests wealth among a pool of assets with an initial
        portfolio b1 and holds the portfolio until the end.

        Parameters:
        -------------
        symbols: list of sting,
            size: n_symbol

        risk_rois : xarray.DataArray,
            dim:(trans_date, symbol),
            shape: (n_period, n_stock)
            The return of all stocks in the given intervals.
            The n_exp_period should be subset of the n_period.

        risk_free_rois : xarray.DataArray,
            dim: (trans_date),
            shape: (n_exp_period, )
            The return of risk-free asset, usually all zeros.

        initial_risk_wealth : xarray.DataArray, shape: (n_stock,)
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
        """
        self.symbols = symbols
        self.n_symbol = len(symbols)
        self.risk_free_symbol = 'risk_free'

        # pandas.core.indexes.datetimes.DatetimeIndex
        self.all_trans_dates = risk_rois.get_index('trans_date')
        self.n_all_period = len(self.all_trans_dates)

        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        # valid number of symbols
        self.valid_dimension("n_symbol", len(symbols),
                             len(initial_risk_wealth))
        self.initial_risk_wealth = initial_risk_wealth
        self.initial_risk_free_wealth = initial_risk_free_wealth

        # verify transaction fee
        self.valid_range_value("buy_trans_fee", buy_trans_fee, 0, 1)
        self.buy_trans_fee = buy_trans_fee

        self.valid_range_value("sell_trans_fee", sell_trans_fee, 0, 1)
        self.sell_trans_fee = sell_trans_fee

        # .loc() will contain the end_date element
        self.valid_trans_date(start_date, end_date)
        self.exp_risk_rois = risk_rois.loc[start_date:end_date]
        self.exp_risk_free_rois = risk_free_rois.loc[start_date:end_date]
        self.n_exp_period = self.exp_risk_rois.shape[0]
        self.n_symbol = self.exp_risk_rois.shape[1]

        # date index in total data
        self.start_date_idx = self.risk_rois.index.get_loc(
            self.exp_risk_rois.index[0])

        # results data
        # risk wealth DataFrame, shape: (n_exp_period, n_stock)
        self.risk_wealth_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_symbol)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # risk_free Series, shape: (n_exp_period, )
        self.risk_free_wealth = pd.Series(np.zeros(self.n_exp_period),
                                          index=self.exp_risk_free_rois.index)

        # buying amount DataFrame, shape: (n_exp_period, n_stock)
        self.buy_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_symbol)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # selling amount DataFrame, shape: (n_exp_period, n_stock)
        self.sell_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_symbol)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # cumulative loss in transaction fee in the simulation
        self.trans_fee_loss = 0

    def get_simulation_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        func_name: str, Function name of the class
        """
        return "BAH_M{}".format(self.n_symbol)

    def run(self):
        """
        run simulation
        Returns:
        ----------------
        standard report
        """
        t0 = time()

        # get function name
        func_name = self.get_trading_func_name()

        # current wealth of each stock in the portfolio
        # at the first period, allocated fund uniformly to each stock
        allocated_risk_wealth = self.initial_risk_wealth
        allocated_risk_free_wealth = self.initial_risk_free_wealth

        for tdx in xrange(self.n_exp_period):
            t1 = time()
            if tdx == 0:
                # the first period, uniformly allocation fund
                # the transaction fee  should be considered while buying
                buy_amounts = pd.Series(
                    np.ones(self.n_symbol) *
                    self.initial_risk_free_wealth / self.n_symbol /
                    (1 + self.buy_trans_fee),
                    index=self.symbols)
                self.buy_amounts_df.iloc[tdx] = buy_amounts
                sell_amounts = 0

                buy_amounts_sum = buy_amounts.sum()
                sell_amounts_sum = 0

            elif tdx == self.n_exp_period - 1:
                # the last period, sell all stocks at the last period
                buy_amounts = 0
                sell_amounts = self.risk_wealth_df.iloc[tdx - 1]

                buy_amounts_sum = 0
                sell_amounts_sum = sell_amounts.sum()

            else:
                buy_amounts, sell_amounts = 0, 0
                buy_amounts_sum, sell_amounts_sum = 0, 0

            # record buy and sell amounts
            self.buy_amounts_df.iloc[tdx] = buy_amounts
            self.sell_amounts_df.iloc[tdx] = sell_amounts

            self.trans_fee_loss += (
                    buy_amounts_sum * self.buy_trans_fee +
                    sell_amounts_sum * self.sell_trans_fee
            )

            total_buy = (buy_amounts_sum * (1 + self.buy_trans_fee))
            total_sell = (sell_amounts_sum * (1 - self.sell_trans_fee))

            # capital allocation
            self.risk_wealth_df.iloc[tdx] = (
                    (1 + self.exp_risk_rois.iloc[tdx]) *
                    allocated_risk_wealth +
                    self.buy_amounts_df.iloc[tdx] - self.sell_amounts_df.iloc[
                        tdx]
            )
            self.risk_free_wealth.iloc[tdx] = (
                    (1 + self.exp_risk_free_rois.iloc[tdx]) *
                    allocated_risk_free_wealth -
                    total_buy + total_sell
            )

            # update wealth
            allocated_risk_wealth = self.risk_wealth_df.iloc[tdx]
            allocated_risk_free_wealth = self.risk_free_wealth.iloc[tdx]

            print("[{}/{}] {} {} OK, "
                  "current_wealth:{:.2f}, {:.3f} secs".format(
                tdx + 1, self.n_exp_period,
                self.exp_risk_rois.index[tdx].strftime("%Y%m%d"),
                func_name,
                (self.risk_wealth_df.iloc[tdx].sum() +
                 self.risk_free_wealth.iloc[tdx]),
                time() - t1))

        # end of iterations, computing statistics
        final_wealth = (self.risk_wealth_df.iloc[-1].sum() +
                        self.risk_free_wealth[-1])

        # get reports
        reports = self.get_performance_report(
            func_name,
            self.symbols,
            self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[-1],
            self.buy_trans_fee,
            self.sell_trans_fee,
            (self.initial_risk_wealth.sum() + self.initial_risk_free_wealth),
            final_wealth,
            self.n_exp_period,
            self.trans_fee_loss,
            self.risk_wealth_df,
            self.risk_free_wealth)

        # model additional elements to reports
        reports['buy_amounts_df'] = self.buy_amounts_df
        reports['sell_amounts_df'] = self.sell_amounts_df

        # add simulation time
        reports['simulation_time'] = time() - t0

        print("{} OK n_stock:{}, [{}-{}], {:.4f}.secs".format(
            func_name, self.n_symbol,
            self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[-1],
            time() - t0))

        return reports


if __name__ == '__main__':
    main()
