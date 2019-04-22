# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: nonecheck=False

"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""
import os
import platform
import sys

import numpy as np
import xarray as xr


cdef class ValidMixin:

    @staticmethod
    cdef valid_exp_name(str exp_name):
        if exp_name not in ('dissertation', 'stocksp'):
            raise ValueError('unknown exp_name:{}'.format(exp_name))

    @staticmethod
    cdef valid_setting(str setting):
        if setting not in ("compact", "general"):
            raise ValueError("Unknown setting: {}".format(setting))


    @staticmethod
    cdef valid_range_value(str name, double value, double lower_bound,
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
    cdef valid_nonnegative_value(str name, double value):
        """
        Parameter:
        -------------
        name: string
            name of the value
        value : integer or float
        """
        if value < 0:
            raise ValueError("The {}'s value {} should be nonnegative.".format(
                name, value))

    @staticmethod
    cdef valid_positive_value(str name, double value):
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
    cdef valid_nonnegative_list(str name, list values):
        """
        Parameter:
        -------------
        name: string
            name of the value
        value : list[int] or list[float]
        """
        arr = np.asarray(values)
        if np.any(arr < 0):
            raise ValueError("The {} contain negative values.".format(
                name, arr))

    @staticmethod
    cdef valid_dimension(dim1_name, dim1, dim2):
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
    cdef valid_trans_date(start_date, end_date):
        """
        Parameters:
        --------------
        start_date, end_date: datetime.date
        """
        if start_date >= end_date:
            raise ValueError("wrong transaction interval, start:{}, "
                             "end:{})".format(start_date, end_date))

if __name__ == '__main__':
    pass