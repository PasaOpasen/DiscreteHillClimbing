
from typing import Callable, Optional, Sequence, Tuple, Union

import collections.abc

import random
import numpy as np


def set_seed(seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def _fix_random_counts(
    random_counts_by_predictors: Union[int, Sequence[int]],
    dim_lengths: Sequence[int],
    dim_total: int
):

    start_type = type(random_counts_by_predictors)

    if start_type == int:
        arr = np.full(dim_total, random_counts_by_predictors)

    elif isinstance(random_counts_by_predictors, collections.abc.Sequence):
        assert len(random_counts_by_predictors) == dim_total, f"'random_counts_by_predictors' must be {dim_total}-dimention or 1-D, not {len(random_counts_by_predictors)}"
        arr = np.array(random_counts_by_predictors)

    else:
        raise Exception(f"type {start_type} if not supported for 'random_counts_by_predictors'; use int or sequence of int")

    for i in range(dim_total):
        arr[i] = min(arr[i], dim_lengths[i])
    
    return arr

def _fix_start_solution(
    available_predictors_values: Sequence[np.ndarray],
    dim_lengths: Sequence[int],
    start_solution: Optional[Sequence[float]] = None
):

    if start_solution is None:
        return np.array([random.randrange(0, dim) for dim in dim_lengths])

    assert isinstance(start_solution, collections.abc.Sequence), f"start_solution must be None or Sequence of float, not {type(start_solution)}"
    assert len(start_solution) == len(dim_lengths), f"start_solution has incorrect lenght {len(start_solution)}, but there are {len(dim_lengths)} predictors of function"

    return np.array([arr.tolist().index(val) for arr, val in zip(available_predictors_values, start_solution)])



def get_best_choice(
    func: Callable[[np.ndarray], float],
    val: float,
    arr: np.ndarray,
    index: int,
    available_values_length: int,
    count: int
) -> Tuple[int, float, int]:
    """
    get best available value from (index)-dimension

    Parameters
    ----------
    func
    val
    arr
    index
    available_values_length
    count

    Returns
    -------

    """
    start_val = arr[index]
    candidates = list(range(0, start_val)) + list(range(start_val + 1, available_values_length))

    if count < available_values_length - 1:
        candidates = random.sample(candidates, count)

    scores = np.empty(len(candidates) + 1)
    start_val = arr[index]

    for i in range(len(candidates)):
        arr[index] = candidates[i]
        scores[i] = func(arr)
        arr[index] = start_val
    
    scores[-1] = val
    candidates.append(start_val)

    min_arg = scores.argmin()

    return (candidates[min_arg], scores[min_arg], len(candidates)-1)





def Hill_Climbing_descent(
    function: Callable[[np.ndarray], float],

    available_predictors_values: Sequence[np.ndarray],
    random_counts_by_predictors: Union[int, Sequence[int]] = 3,
    greedy_step: int = 1,

    start_solution: Optional[Sequence[float]] = None,
    max_function_evals: int = 1000,
    maximize: bool = False,

    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    function which makes Hill Climbing descent

    Parameters
    ----------
    function : func np.array->float/int
        callable optimized function uses numpy 1D-array as argument.
    available_predictors_values : list of numpy 1D-arrays
        a list of available values for each predictor (each dimention of argument).
    random_counts_by_predictors : int or int sequence array, optional
        how many random choices should it use for each variable? Use list/numpy array for select option for each predictor (or int -- one number for each predictor). The default is 3.
    greedy_step : int, optional
        it choices better solution after climbing by greedy_step predictors. The default is 1.
    start_solution : Optional[Sequence[float]]
        point when the algorithm starts. The default is None, means random point.
    max_function_evals : int, optional
        max count of function evaluations. The default is 1000.
    maximize : bool, optional
        maximize the function? (minimize by default). The default is False.

    seed : int or None
        random seed (None if doesn't matter). The default is None

    Returns
    -------
    best_pos : np array
        best solution.
    best_val : float
        best score.

    """
    
    set_seed(seed)
    assert len(available_predictors_values) > 0

    inds_to_vals = lambda inds: np.array([arr[ind] for arr, ind in zip(available_predictors_values, inds)])
    func = (lambda x: -function(inds_to_vals(x))) if maximize else (lambda x: function(inds_to_vals(x)))

    dim_lengths = np.array([len(arr) for arr in available_predictors_values])
    dim_total = len(available_predictors_values)

    assert 1 <= greedy_step <= dim_total, f"greedy_step ({greedy_step}) should be in [1, {dim_total}]"

    predictor_indexes = np.arange(dim_total)
    counts_by_predictors = _fix_random_counts(random_counts_by_predictors, dim_lengths, dim_total)

    start_pos = _fix_start_solution(available_predictors_values, dim_lengths, start_solution)
    start_val = func(start_pos)

    best_pos = start_pos.copy()
    best_val = start_val

    func_evals = 1

    while func_evals < max_function_evals:

        pred_indexes = np.random.choice(predictor_indexes, greedy_step, replace=False)

        candidates = [
            get_best_choice(func, best_val, best_pos, predictor_index, dim_lengths[predictor_index], counts_by_predictors[predictor_index])
            for predictor_index in pred_indexes
        ]

        scores = np.array([t[1] for t in candidates])

        arg_min = scores.argmin()

        best_val = scores[arg_min]
        best_pos[pred_indexes[arg_min]] = candidates[arg_min][0]

        func_evals += sum((t[2] for t in candidates))
    
    if maximize:
        best_val *= -1

    return (inds_to_vals(best_pos), best_val)




if __name__ == '__main__':

    f = lambda X: np.sum(X) - X[0]*X[2]

    bounds = [
        np.arange(100)+100,
        np.arange(100),
        np.arange(100),
        np.arange(100),
        np.arange(100),
        np.arange(100),
        np.arange(100)
    ]

    v = Hill_Climbing_descent(f, bounds, random_counts_by_predictors = 10, greedy_step=2, max_function_evals= 10000, maximize= True)

    print(v)






















