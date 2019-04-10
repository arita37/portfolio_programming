# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import sys
from time import time
import numpy as np


def swap_regret():
    size = 5
    weights = np.arange(6, 1, -1)
    experts = np.tile(weights, (size * (size - 1), 1))
    print(experts)
    row = 0
    for idx in range(size):
        for jdx in range(size):
            if idx != jdx:
                print(idx, jdx, row)
                experts[row, jdx] += experts[row, idx]
                experts[row, idx] = 0
                row += 1
    print(experts)


def ir_example(n_expert=3):
    # p = np.array([0.2, 0.3, 0.5])
    p = np.random.dirichlet(np.random.rand(n_expert))

    virtual_experts = modified_probabilities(p)

    # rel = np.array([1.05, 0.9, 0.98])
    rel = np.random.rand(n_expert) + 1
    new_weights = np.exp(np.log((virtual_experts * rel).sum(axis=1)))
    virtual_expert_weights = new_weights / new_weights.sum()
    print(virtual_expert_weights)
    # A = np.array([
    #     [-virtual_expert_weights[0] - virtual_expert_weights[1],
    #      virtual_expert_weights[0],
    #      virtual_expert_weights[1]
    #      ],
    #     [virtual_expert_weights[2],
    #      -virtual_expert_weights[2] - virtual_expert_weights[3],
    #      virtual_expert_weights[3]
    #      ],
    #     [virtual_expert_weights[4],
    #      virtual_expert_weights[5],
    #      -virtual_expert_weights[4] - virtual_expert_weights[5]
    #      ]
    # ])
    #
    # # print('*'*10 + 'col stochastic matrix')
    # Q = A.T / np.max(np.abs(A)) + np.identity(3)
    Q = column_stochastic_matrix(n_expert, virtual_expert_weights)
    # print(Q)
    # print(Q.sum(axis=0))
    eigs2, eigvs2 = np.linalg.eig(Q)
    # print('eigen values:', eigs2)
    # print("eigen vector:", eigvs2)

    prob = (eigvs2[:, 0] / eigvs2[:, 0].sum()).astype(np.float64)
    print(prob, prob.sum())

    experts2 = np.tile(prob, (n_expert * (n_expert - 1), 1))
    row = 0
    for idx in range(n_expert):
        for jdx in range(n_expert):
            if idx != jdx:
                # print(idx, jdx, row)
                experts2[row, jdx] += experts2[row, idx]
                experts2[row, idx] = 0
                row += 1
    # print(experts2)

    result = np.zeros(n_expert)
    for idx in range(n_expert * (n_expert-1)):
        result += experts2[idx, :] * virtual_expert_weights[idx]
    # print(result)

    # print("prob:", prob, prob.sum())
    if not np.allclose(prob.sum(), 1) and not np.allclose(prob, result):
        sys.exit(1)


def modified_probabilities(probs):
    """
    Parameters:
    ------------
    probs: array like
        shape: n_action

    Returns:
    -------------------
    size * (size - 1) modified probabilities
    """
    n_action = len(probs)
    virtual_experts = np.tile(probs, (n_action * (n_action - 1), 1))
    row = 0
    for idx in range(n_action):
        for jdx in range(n_action):
            if idx != jdx:
                virtual_experts[row, jdx] += virtual_experts[row, idx]
                virtual_experts[row, idx] = 0
                row += 1
    return virtual_experts


def column_stochastic_matrix(n_action, virtual_expert_weights):
    """
    Parameters:
    ------------
    n_action: int, number of actions
    virtual_expert_weights: array like,
        shape: n_action * (n_action-1)

    Returns:
    -------------
    numpy.array, shape: n_action * n_action
    """
    n_virtual_expert = len(virtual_expert_weights)
    assert n_virtual_expert == n_action * (n_action - 1)

    # build row-sum-zero matrix
    A = np.insert(virtual_expert_weights,
                  np.arange(0, n_action * n_action, n_action),
                  np.zeros(n_action)).reshape((n_action, n_action))
    np.fill_diagonal(A, -A.sum(axis=1))

    # column stochastic matrix
    S = A.T / np.max(np.abs(A)) + np.identity(n_action)
    return S


def run_column_stochastic_matrix():
    n_action = 5
    virtual_expert_weights = np.random.dirichlet(
        np.random.rand(n_action * (n_action - 1)))
    print(virtual_expert_weights)
    print(virtual_expert_weights.sum())
    column_stochastic_matrix(n_action, virtual_expert_weights)


if __name__ == '__main__':
    # swap_regret()
    ir_example(10)
    # for _ in range(50000):
    #     ir_example()
    # run_column_stochastic_matrix()
