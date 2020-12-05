
import random
import numpy as np



def correct_random_counts(random_counts_by_predictors, dim_lengths, dim_total):

    start_type = type(random_counts_by_predictors)

    if start_type == int:
        arr = np.full(dim_total, random_counts_by_predictors)
    elif start_type == list:
        if len(random_counts_by_predictors) != dim_total:
            raise Exception(f"'random_counts_by_predictors' must be {dim_total}-dimention or 1-D, not {len(random_counts_by_predictors)}")
        arr = np.array(random_counts_by_predictors)
    elif start_type == np.ndarray:
        if random_counts_by_predictors.size != dim_total:
            raise Exception(f"'random_counts_by_predictors' must be {dim_total}-dimention or 1-D, not {random_counts_by_predictors.size}")
        arr = random_counts_by_predictors.copy()
    else:
        raise Exception(f"type {start_type} if not supported for 'random_counts_by_predictors'; use int, list or numpy array")

    for i in range(dim_total):
        arr[i] = min(arr[i], dim_lengths[i])
    
    return arr

def correct_start_solution(available_predictors_values, dim_lengths, start_solution):

    if start_solution == 'random':
        return np.array([random.randrange(0, dim) for dim in dim_lengths])
    
    if not (type(start_solution) in (list, np.ndarray)):
        raise Exception(f"incorrect type {type(start_solution)} of start solution; should be list, numpy array or 'random'")

    if len(start_solution) != len(dim_lengths):
        raise Exception("start_solution has incorrect lenght {len(start_solution)}, but there are {len(dim_lengths)} predictors of function")

    return np.array([arr.to_list().index(val) for arr, val in zip(available_predictors_values, start_solution)])



def get_best_choice(func, val, arr, index, available_values_length, count):
    inds = list(range(0, arr[index])) + list(range(arr[index], available_values_length))

    if count < available_values_length - 1:
        inds = random.sample(inds, count)

    scores = np.empty(len(inds) + 1)
    start_val = arr[index]

    for i in range(len(inds)):   
        arr[index] = inds[i]
        scores[i] = func(arr)
        arr[index] = start_val
    
    scores[-1] = val
    inds.append(arr[index])

    min_arg = np.argmin(scores)

    return (inds[min_arg], scores[min_arg], len(inds)-1)





def Hill_Climbing_descent(function, available_predictors_values, random_counts_by_predictors = 3, greedy_step = 1, start_solution = 'random', max_function_evals = 1000, maximize = False, seed = None):
    """
    function which makes Hill Climbing descent

    Parameters
    ----------
    function : func np.array->float/int
        callable optimized function uses numpy 1D-array as argument.
    available_predictors_values : list of numpy 1D-arrays
        a list of available values for each predictor (each dimention of argument).
    random_counts_by_predictors : int/list/numpy array, optional
        how many random choices should it use for each variable? Use list/numpy array for select option for each predictor (or int -- one number for each predictor). The default is 3.
    greedy_step : int, optional
        it choices better solution after climbing by greedy_step predictors. The default is 1.
    start_solution : 'random' or list or np array, optional
        point when the algorithm starts. The default is 'random'.
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
    
    if not (seed is None):
        np.random.seed(seed)
        random.seed(seed)

    inds_to_vals = lambda inds: np.array([arr[ind] for arr, ind in zip(available_predictors_values, inds)])

    func = (lambda x: -function(inds_to_vals(x))) if maximize else (lambda x: function(inds_to_vals(x)))

    dim_lengths = np.array([len(arr) for arr in available_predictors_values])
    dim_total = len(dim_lengths)
    if greedy_step > dim_total or greedy_step < 1:
        raise Exception(f"greedy_step ({greedy_step}) should be in [1, {dim_total}]")
    predictor_indexes = np.arange(dim_total)

    counts_by_predictors = correct_random_counts(random_counts_by_predictors, dim_lengths, dim_total)

    start_pos = correct_start_solution(available_predictors_values, dim_lengths, start_solution)
    start_val = func(start_pos)

    best_pos = start_pos.copy()
    best_val = start_val

    func_evals = 1

    while func_evals < max_function_evals:

        pred_indexes = np.random.choice(predictor_indexes, greedy_step, replace=False)

        candidates = [get_best_choice(func, best_val, best_pos, predictor_index, dim_lengths[predictor_index], counts_by_predictors[predictor_index]) for predictor_index in pred_indexes]

        scores = np.array([t[1] for t in candidates])

        arg_min = np.argmin(scores)

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






















