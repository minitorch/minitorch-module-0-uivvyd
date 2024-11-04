"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplies two numbers"""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers"""
    return x + y


def neg(x: float) -> float:
    """Negates a number"""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another"""
    return float(x < y)


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal"""
    return float(x == y)


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value"""
    return float(abs(x - y) < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -y / x ** 2


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return y * (x > 0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable"""
    return lambda arr: [fn(x) for x in arr]


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function"""

    def zipWith_tmp(arr1, arr2):
        return [fn(x1, x2) for x1, x2 in zip(arr1, arr2)]

    return zipWith_tmp


def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function"""

    def reduce_tmp(arr):
        res = start
        for x in arr:
            res = fn(res, x)
        return res

    return reduce_tmp


def negList(arr: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    return map(neg)(arr)


def addLists(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add)(arr1, arr2)


def sum(arr: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, 0.0)(arr)


def prod(arr: Iterable[float]) -> float:
    """ Calculate the product of all elements in a list using reduce"""
    return reduce(mul, 1.0)(arr)
