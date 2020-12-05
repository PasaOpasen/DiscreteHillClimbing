import sys
sys.path.append('..')

import numpy as np
from DiscreteHillClimbing import Hill_Climbing_descent


def func(array):
    return (np.sum(array) + np.sum(array**2))/(1 + array[0]*array[1])


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


solution, value = Hill_Climbing_descent(function = func,
    available_predictors_values = available_predictors_values,
    random_counts_by_predictors = [4, 5, 2, 20, 20, 3, 6, 6, 4],
    greedy_step = 1,
    start_solution = 'random',
    max_function_evals = 1000,
    maximize = False,
    seed = 1)

print(solution)
print(value)

# [  10.   -5.  494.  100.  112.    9.  123.    9. 1000.]
# -26200.979591836734