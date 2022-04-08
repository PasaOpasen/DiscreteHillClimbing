# Discrete Hill Climbing

[![PyPI
version](https://badge.fury.io/py/DiscreteHillClimbing.svg)](https://pypi.org/project/DiscreteHillClimbing/)
[![Downloads](https://pepy.tech/badge/DiscreteHillClimbing)](https://pepy.tech/project/DiscreteHillClimbing)
[![Downloads](https://pepy.tech/badge/DiscreteHillClimbing/month)](https://pepy.tech/project/DiscreteHillClimbing)
[![Downloads](https://pepy.tech/badge/DiscreteHillClimbing/week)](https://pepy.tech/project/DiscreteHillClimbing)

This is the implementation of Hill Climbing algorithm for discrete tasks.

```
pip install DiscreteHillClimbing
```

- [Discrete Hill Climbing](#discrete-hill-climbing)
  - [About](#about)
  - [Why Hill Climbing?](#why-hill-climbing)
  - [Pseudocode](#pseudocode)
  - [Working process](#working-process)
  - [Parameters](#parameters)
  - [Examples](#examples)
    - [Own random search counts](#own-random-search-counts)
    - [Different greedy steps](#different-greedy-steps)

## About

[Hill Climbing](https://en.wikipedia.org/wiki/Hill_climbing) (*coordinate minimization*) is the most simple algorithm for discrete tasks a lot (one simpler is only getting best from fully random). In discrete tasks each predictor can have it's value from finite set, therefore we can check all values of predictor (variable) or some not small random part of it and do optimization by one predictor. After that we can optimize by another predictor and so on. Also we can try to find better solution using 1, 2, 3, ... predictors and choose only the best configuration. 

There is a highly variety of ways to realize **hill climbing**, so I tried to make just simple and universal implementation. Assuredly, it can be better to create your (faster)implementation depends on specific of ur own task, but this package is a good flexible choice for start.

## Why Hill Climbing?

Hill Climbing is the perfect baseline when u start to seek solution. It really helps and it really can get u very good result using just 50-200 function evaluations.

## Pseudocode

See the main idea of my implementation in this pseudocode:

```c
best solution <- start_solution
best value <- func(start_solution)

while functions evaluates_count < max_function_evals:
    predictors <- get greedy_step random predictors (variables)
    for each predictor from predictors:
        choose random_counts_by_predictors values from available values of this predictor
        for each chosen value:
            replace predictor value with chosen
            evaluate function
        remember best result as result of this predictor
    select best predictor with its best configuration
    replace values in solution
    it is new best solution 
```

## Working process

**Load packages**:
```python
import numpy as np
from DiscreteHillClimbing import Hill_Climbing_descent
```

**Determine optimized function**:
```python
def func(array: np.ndarray) -> float:
    return (np.sum(array) + np.sum(array**2))/(1 + array[0]*array[1])
```

**Determine available sets for each predictor**:
```python
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
```

**Create solution**:
```python
solution, value = Hill_Climbing_descent(function = func,
    available_predictors_values = available_predictors_values,
    random_counts_by_predictors = 4,
    greedy_step = 1,
    start_solution = None,
    max_function_evals = 1000,
    maximize = False,
    seed = 1)

print(solution)
print(value)

# [  10.           -5.          493.           98.56521739  112.    9.          123.            9.         1000.        ]
# -26174.972801975237
```

## Parameters

* `function` : func np.array->float/int; callable optimized function uses numpy 1D-array as argument.
* `available_predictors_values` : int sequence
        a list of available values for each predictor (each dimention of argument).
* `random_counts_by_predictors` : int or int sequence, optional
        how many random choices should it use for each variable? Use list/numpy array for select option for each predictor (or int -- one number for each predictor). The default is 3.
* `greedy_step` : int, optional
        it choices better solution after climbing by greedy_step predictors. The default is 1.
* `start_solution` : None or int sequence, optional
        point when the algorithm starts. The default is None -- random start solution
* `max_function_evals` : int, optional
        max count of function evaluations. The default is 1000.
* `maximize` : bool, optional
        maximize the function? (minimize by default). The default is False.

* `seed` : int or None. Random seed (None if doesn't matter). The default is None

**Returns**: tuple contained best solution and best function value.

## Examples

### Own random search counts

U can set your different `random_counts_by_predictors` value for each predictor. For example, if the predictor available set contains only 3-5 values, it's better to check all of them; but if there are 100+ values, it will be better to check 20-40 of them, not 5. See example:

```python
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
    start_solution = None,
    max_function_evals = 1000,
    maximize = False,
    seed = 1)

print(solution)
print(value)

# [  10.   -5.  494.  100.  112.    9.  123.    9. 1000.]
# -26200.979591836734
```

### Different greedy steps

Parameter `greedy_step` can be important in some tasks. It can be better to use `greedy_step = 2` or `greedy_step = 3` instead of default `greedy_step = 1`. And it's not good choice to use big `greedy_step` if we cannot afford to evaluate optimized function many times.

```python
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

print(pd.DataFrame({'greedy_step': greedys, 'result': results}).sort_values(['result'], ascending=False))

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
```

