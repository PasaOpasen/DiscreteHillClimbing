import sys
sys.path.append('..')

import numpy as np
from DiscreteHillClimbing import Hill_Climbing_descent


def func(array):
    return (np.sum(array) + np.sum(np.abs(array)))/(1 + array[0]*array[1])-array[3]*array[4]*(25-array[5])


available_predictors_values = [
    np.array([10, 20, 35, 50]),
    np.array([-5, 3, -43, 0, 80]),
    np.arange(40, 500),
    np.linspace(1, 100, num = 70),
    np.array([65, 32, 112]),
    np.array([1, 2, 3, 0, 9, 8]),
    np.array([1, 11, 111, 123, 43]),
    np.array([8, 9, 0, 5, 4, 3]),
    np.array([-1000, -500, 500, 1000])
]


seeds = np.arange(100)

greedys = [1, 2, 3, 4, 5, 6, 7, 8, 9]

results = np.empty(len(greedys))

for i, g in enumerate(greedys):
    vals = []
    for s in seeds:
        _, value = Hill_Climbing_descent(function = func,
            available_predictors_values = available_predictors_values,
            random_counts_by_predictors = [4, 5, 2, 20, 20, 3, 6, 6, 4],
            greedy_step = g,
            start_solution = [v[0] for v in available_predictors_values],
            max_function_evals = 100,
            maximize = True,
            seed = s)
        vals.append(value)
    
    results[i] = np.mean(vals)

import pandas as pd

print(
    pd.DataFrame(
        {'greedy_step': greedys, 'result': results}
    ).sort_values(
        ['result'],
        ascending=False
     )
)

#    greedy_step       result
# 1            2  1634.172483
# 0            1  1571.038514
# 2            3  1424.222610
# 3            4  1320.051325
# 4            5  1073.783527
# 5            6   847.873058
# 6            7   362.113555
# 7            8    24.729801
# 8            9  -114.200000