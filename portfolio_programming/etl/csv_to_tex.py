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


def BAH_to_latex(exp_name):
    if exp_name == 'dissertation':
        csv_tex_files = [
            [
                os.path.join(pp.TMP_DIR, "BAH_stat.csv"),
                os.path.join(pp.TMP_DIR, "BAH_stat.txt"),
            ]
        ]
        # ['simulation_name', 'group_name', 'start_date', 'end_date',
        # 'n_data', 'cum_roi', 'annual_roi', 'roi_mu', 'std', 'skew',
        # 'ex_kurt', 'Sharpe', 'Sortino_full', 'Sortino_partial',
        # 'SPA_c']

        line_style = "{:>6}  & {:>8.2f} & {:>8.2f} & {:>10.4f} & " \
                     "{:>6.4f} & {:>6.2f} & {:6.2f} & {:6.2f} & {:6.2f} " \
                     "& {:>8}  \\\\ \\hline \n"

        for csv_file, tex_file in csv_tex_files:
            rows = csv.DictReader(open(csv_file))
            with open(tex_file, 'w') as fout:
                for row in rows:
                    fout.writelines(line_style.format(
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
    BAH_to_latex('dissertation')
