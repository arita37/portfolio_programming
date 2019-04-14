# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chen1116@gmail.com>

transforming csv results to latex format
"""

import csv
import os
import portfolio_programming as pp


def symbol_statistics_to_latex(exp_name):
    if exp_name == 'dissertation':
        csv_tex_files = [
            [
                os.path.join(pp.TMP_DIR, "DJIA_2005_symbols_stat.csv"),
                os.path.join(pp.TMP_DIR, "DJIA_2005_symbols_stat.txt")
            ],
            [
                os.path.join(pp.TMP_DIR, "TAIEX_2005_market_cap_stat.csv"),
                os.path.join(pp.TMP_DIR, "TAIEX_2005_market_cap_stat.txt"),
            ]
        ]

        # ['rank', 'group', 'symbol', 'start_date', 'end_date', 'n_data',
        # 'cum_roi', 'annual_roi', 'roi_mu', 'std', 'skew', 'ex_kurt',
        # 'Sharpe', 'Sortino', 'SPA_c', 'JB', 'worst_ADF']

        line_style = "{:>6} & {:>4} & {:>8.2f} & {:>8.2f} & {:>10.4f} & " \
                     "{:>6.4f} & {:>6.2f} & {:6.2f} & {:6.2f} & {:6.2f} &  " \
                     "{:>8} & {:>8} & {:>8} \\\\ \\hline \n"
        for csv_file, tex_file in csv_tex_files:
            rows = csv.DictReader(open(csv_file))
            with open(tex_file, 'w') as fout:
                for row in rows:
                    fout.writelines(line_style.format(
                        row['group'],
                        row['symbol'],
                        float(row['cum_roi']) * 100,
                        float(row['annual_roi']) * 100,
                        float(row['roi_mu']) * 100,
                        float(row['std']) * 100,
                        float(row['skew']),
                        float(row['ex_kurt']),
                        float(row['Sharpe']) * 100,
                        float(row['Sortino']) * 100,
                        star_p_value(float(row['SPA_c']) * 100),
                        star_p_value(float(row['JB']) * 100),
                        star_p_value(float(row['worst_ADF']) * 100)
                    ))
            print("{} complete".format(csv_file))
    else:
        raise ValueError("unknown exp_name:{}".format(exp_name))


def market_index_statistics_to_latex(exp_name):
    if exp_name == 'dissertation':
        csv_tex_files = [
            [
                os.path.join(pp.TMP_DIR, "market_index_stat.csv"),
                os.path.join(pp.TMP_DIR, "market_index_stat.txt")
            ],
        ]

        # ['symbol', 'start_date', 'end_date', 'n_data',
        # 'cum_roi', 'annual_roi', 'roi_mu', 'std', 'skew', 'ex_kurt',
        # 'Sharpe', 'Sortino', 'SPA_c', 'JB', 'worst_ADF']

        line_style = "{:>4} & {:>8.2f} & {:>8.2f} & {:>10.4f} & " \
                     "{:>6.4f} & {:>6.2f} & {:6.2f} & {:6.2f} & {:6.2f} &  " \
                     "{:>8} & {:>8} & {:>8} \\\\ \\hline \n"
        for csv_file, tex_file in csv_tex_files:
            rows = csv.DictReader(open(csv_file))
            with open(tex_file, 'w') as fout:
                for row in rows:
                    fout.writelines(line_style.format(

                        row['symbol'],
                        float(row['cum_roi']) * 100,
                        float(row['annual_roi']) * 100,
                        float(row['roi_mu']) * 100,
                        float(row['std']) * 100,
                        float(row['skew']),
                        float(row['ex_kurt']),
                        float(row['Sharpe']) * 100,
                        float(row['Sortino']) * 100,
                        star_p_value(float(row['SPA_c']) * 100),
                        star_p_value(float(row['JB']) * 100),
                        star_p_value(float(row['worst_ADF']) * 100)
                    ))
            print("{} complete".format(csv_file))
    else:
        raise ValueError("unknown exp_name:{}".format(exp_name))

def strategy_to_latex(exp_name, strategy='eg'):
    if exp_name == 'dissertation':

        # ['simulation_name', 'group_name', 'start_date', 'end_date',
        # 'n_data', 'cum_roi', 'annual_roi', 'roi_mu', 'std', 'skew',
        # 'ex_kurt', 'Sharpe', 'Sortino_full', 'Sortino_partial',
        # 'SPA_c']
        if strategy == 'BAH':
            csv_tex_files = [
                [
                    os.path.join(pp.TMP_DIR, "BAH_stat.csv"),
                    os.path.join(pp.TMP_DIR, "BAH_stat.txt"),
                ],
            ]

        elif strategy in ('eg', 'exp', 'nofee_eg', 'nofee_exp', 'poly',
                          'nofee_poly', 'b1exp', 'nofee_b1exp', 'b1pol',
                          'nofee_b1pol'):
            csv_tex_files = [
                [
                    os.path.join(pp.TMP_DIR, "{}_stat.csv".format(strategy)),
                    os.path.join(pp.TMP_DIR, "{}_stat.txt".format(strategy)),
                ],
            ]

        line_style = "{:>6.2f} & {:>6} & {:>8.2f} & {:>8.2f} & {:>10.4f} & " \
                     "{:>6.4f} & {:>6.2f} & {:6.2f} & {:6.2f} & {:6.2f} " \
                     "& {:>8}  \\\\ \\hline \n"

        for csv_file, tex_file in csv_tex_files:
            rows = csv.DictReader(open(csv_file))
            with open(tex_file, 'w') as fout:
                for row in rows:
                    print(row)
                    if strategy == 'BAH':
                        param = 0
                    elif strategy in ('eg', 'exp', 'nofee_eg', 'nofee_exp',
                                      'b1exp', 'nofee_b1exp'):
                        param = float(row['eta'])
                    elif strategy in ('poly', 'nofee_poly', 'b1pol',
                                      'nofee_b1pol'):
                        param = float(row['poly_power'])
                    fout.writelines(line_style.format(
                        param,
                        row['group_name'],
                        float(row['cum_roi']) * 100,
                        float(row['annual_roi']) * 100,
                        float(row['roi_mu']) * 100,
                        float(row['std']) * 100,
                        float(row['skew']),
                        float(row['ex_kurt']),
                        float(row['Sharpe']) * 100,
                        float(row['Sortino_full']) * 100,
                        star_p_value(float(row['SPA_c']) * 100),
                    ))
            print("{} complete".format(csv_file))
    else:
        raise ValueError("unknown exp_name:{}".format(exp_name))



def spsp_strategy_to_latex(exp_name, strategy='spsp'):
    if exp_name == 'dissertation':

        #group	ALPHA	rolling_window_size	cum_roi	annual_roi	daily_mean_roi
        # daily_std_roi	daily_skew_roi	daily_ex-kurt_roi	Sharpe	Sortino_full	SPA_c

        if strategy == 'spsp':
            csv_tex_files = [
                [
                    os.path.join(pp.TMP_DIR, "spsp_best_3.csv"),
                    os.path.join(pp.TMP_DIR, "spsp_best_3.txt"),
                ],
            ]

        line_style = "{:>8} & {:>8} & {:>8.2f} & {:>8.2f} & {:>10.4f} & " \
                     "{:>6.4f} & {:>6.2f} & {:6.2f} & {:6.2f} & {:6.2f} " \
                     "& {:>8}  \\\\ \\hline \n"

        for csv_file, tex_file in csv_tex_files:
            rows = csv.DictReader(open(csv_file))
            with open(tex_file, 'w') as fout:
                for row in rows:
                    print(row)
                    fout.writelines(line_style.format(
                        row['group'],
                        "({}, {:.0f}\\%)".format(row['rolling_window_size'],
                                              float(row['alpha'] )* 100),
                        float(row['cum_roi']) * 100,
                        float(row['annual_roi']) * 100,
                        float(row['daily_mean_roi']) * 100,
                        float(row['daily_std_roi']) * 100,
                        float(row['daily_skew_roi']),
                        float(row['daily_ex-kurt_roi']),
                        float(row['Sharpe']) * 100,
                        float(row['Sortino_full']) * 100,
                        star_p_value(float(row['SPA_c']) * 100),
                    ))
            print("{} complete".format(csv_file))
    else:
        raise ValueError("unknown exp_name:{}".format(exp_name))

def star_p_value(p_value):
    if p_value <= 1:
        return "***{:6.2f}".format(p_value)
    elif p_value <= 5:
        return "**{:6.2f}".format(p_value)
    elif p_value <= 10:
        return "*{:6.2f}".format(p_value)
    else:
        return "{:6.2f}".format(p_value)


if __name__ == '__main__':
    # symbol_statistics_to_latex('dissertation')
    # market_index_statistics_to_latex('dissertation')
    # strategy_to_latex('dissertation', 'nofee_b1pol')
    spsp_strategy_to_latex('dissertation', strategy='spsp')