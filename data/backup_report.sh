#!/bin/bash
rsync -avzh --progress -e 'ssh -p 1111' /home/chenhh/workspace_pycharm/portfolio_programming/data/report/ chenhh@localhost:/home/chenhh/experiment_data/SPSP_CVaR/report/
rsync -avzh --progress -e 'ssh -p 1111' /home/chenhh/workspace_pycharm/portfolio_programming/data/scenario/ chenhh@localhost:/home/chenhh/experiment_data/SPSP_CVaR/scenario/

