# Discrete Hill Climbing

This is the implementation of Hill Climbing algorithm for discrete tasks.

- [Discrete Hill Climbing](#discrete-hill-climbing)
  - [About](#about)
  - [Why Hill Climbing?](#why-hill-climbing)
  - [Pseudocode](#pseudocode)
  - [Working process](#working-process)
  - [Parameters](#parameters)
  - [Returns](#returns)
  - [Examples](#examples)

## About

[Hill Climbing](https://en.wikipedia.org/wiki/Hill_climbing) is the most simple algorithm for discrete tasks a lot. In discrete tasks each predictor can have it's value from finite set, therefore we can check all values of predictor or some not small random part of it and do optimization by one predictor. After that we can optimize by another predictor and so on. Also we can try to find better solution using 1, 2, 3, ... predictors and choose only the best configuration. 

There is a highly variety of ways to realize **hill climbing**, so I tried to make just simple and universal implementation. Assuredly, it can be better to create your implementation for your own task, but this package is a good choice for start.

## Why Hill Climbing?

Hill Climbing is the prefect baseline when u start to seek solution. It really helps and it really can get u very good result using just 50-200 function evaluations.

## Pseudocode

See the main idea of my implementation in this pseudocode:

```ps
best solution <- start_solution
best value <- func(start_solution)

while functions evaluates_count < max_function_evals:
    predictors <- get greedy_step random predictors
    for each predictor from predictors:
        choose random_counts_by_predictors values from available values for this predictor
        for each chosen value:
            replace predictor value with chosen
            evaluate function
        remember best result for this predictor
    select best predictor with its best configuration
    replace values in solution
    it is new best solution 
```

## Working process

**Load packages**:
```python

```

**Determine optimized function**:
```python

```

**Determine available sets for each predictor**:
```python

```

**Create solution**:
```python

```

## Parameters

* `function` : func np.array->float/int; callable optimized function uses numpy 1D-array as argument.
* `available_predictors_values` : list of numpy 1D-arrays
        a list of available values for each predictor (each dimention of argument).
* `random_counts_by_predictors` : int/list/numpy array, optional
        how many random choices should it use for each variable? Use list/numpy array for select option for each predictor (or int -- one number for each predictor). The default is 3.
* `greedy_step` : int, optional
        it choices better solution after climbing by greedy_step predictors. The default is 1.
* `start_solution` : 'random' or list or np array, optional
        point when the algorithm starts. The default is 'random'.
* `max_function_evals` : int, optional
        max count of function evaluations. The default is 1000.
* `maximize` : bool, optional
        maximize the function? (minimize by default). The default is False.

## Returns
Tuple contained best solution and best function value.

## Examples
