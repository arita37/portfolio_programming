# -*- coding: utf-8 -*-
#!python
# cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: nonecheck=False

"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import numpy as np
cimport numpy as cnp
from cpython.datetime cimport date

cpdef tuple get_valid_exp_name():
    return ('dissertation', 'stocksp')

cpdef tuple get_valid_setting():
    return ("compact", "general")

cdef class ValidMixin:
    @staticmethod
    cdef void valid_exp_name(str exp_name):
        if exp_name not in get_valid_exp_name():
            raise ValueError('unknown exp_name:{}'.format(exp_name))

    @staticmethod
    cdef void valid_setting(str setting):
        if setting not in get_valid_setting():
            raise ValueError("Unknown setting: {}".format(setting))

    @staticmethod
    cdef void valid_range_value(str name, double value, double lower_bound,
                                double upper_bound):
        """
        Parameter:
        -------------
        name: string
            name of the value
        value : float or integer
        upper bound : float or integer
        lower_bound : float or integer
        """
        if not lower_bound <= value <= upper_bound:
            raise ValueError("The {}' value {} not in the given bound ({}, "
                             "{}).".format(name, value, upper_bound,
                                           lower_bound))

    @staticmethod
    cdef void valid_nonnegative_value(str name, double value):
        """
        Parameter:
        -------------
        name: string
            name of the value
        value : integer or float
        """
        if value < 0:
            raise ValueError("The {}'s value {} should be >=0.".format(
                name, value))

    @staticmethod
    cdef void valid_positive_value(str name, double value):
        """
        Parameter:
        -------------
        name: string
            name of the value

        value : integer or float
        """
        if value <= 0:
            raise ValueError("The {}'s value {} should be positive.".format(
                name, value))

    @staticmethod
    cdef void valid_nonnegative_list(str name, list values):
        """
        Parameter:
        -------------
        name: string
            name of the value
        value : list[int] or list[float]
        """
        arr = cnp.asarray(values)
        if cnp.any(arr < 0):
            raise ValueError("The {} contain negative values.".format(
                name, arr))

    @staticmethod
    cdef void valid_dimension(str dim1_name, int dim1, int dim2):
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
    cdef void valid_trans_date(date start_date, date end_date):
        """
        Parameters:
        --------------
        start_date, end_date: datetime.date
        """
        if start_date >= end_date:
            raise ValueError("wrong transaction interval, start:{}, "
                             "end:{})".format(start_date, end_date))


cdef class SPSPBase(ValidMixin):
    cdef:
        # user specifies
        str setting
        str group_name
        list candidate_symbols
        int max_portfolio_size
        double initial_risk_free_wealth
        double buy_trans_fee, sell_trans_fee
        date start_date, end_date
        int n_scenario
        int scenario_set_idx
        int print_interval
        str report_dir

        # from data
        int n_symbol

    def __cinit__(self,
                 str setting,
                 str group_name,
                 list candidate_symbols,
                 int max_portfolio_size,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=0.001425,
                 double sell_trans_fee=0.004425,
                 start_date=date(2005,1,3),
                 end_date=date(2018,12,28),
                 int rolling_window_size=200,
                 int n_scenario=200,
                 int scenario_set_idx=1,
                 int print_interval=10,
                 str report_dir=""):
        self.setting = setting
        self.group_name = group_name
        self.candidate_symbols = candidate_symbols
        # self.valid_dimension("n_symbol", len(candidate_symbols),
        #                      risk_rois.shape[1])
        self.n_symbol = len(candidate_symbols)
        self.risk_rois =  risk_rois
        self.risk_free_rois = risk_free_rois
        self.initial_risk_free_wealth = initial_risk_free_wealth
        self.buy_trans_fee = buy_trans_fee
        self.sell_trans_fee = sell_trans_fee
        self.exp