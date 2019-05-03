from math import log
import itertools
import collections


def singleinf(x):
    """ Calculate a single term of average entropy"""
    return -x * log(x, 2)


def probability(LT, as_fraction=False):
    """
    This function calculates probability of single events
    or joint probability of multiple events in data series
    (a procedure also known as Maximum Likelihood (ML)
    probabilities or frequencies estimation).
    As input LT we expect an array or a matrix [m x n],
    where m = samples nr and n = nr of samples variables(or dimensions).
    LT should be shaped ([var1,var2,...,varn],
                         ...[i-th sample],...).
    Set as_fraction=True to get fractions in reduced form
    (slower than floats)
    """
    probdict = collections.Counter(LT)
    l = len(LT)
    if as_fraction:
        from fractions import Fraction
        return tuple(Fraction(v, l) for v in probdict.values())
    return tuple(v / l for v in probdict.values())


def entropy(L, k=1):
    """
    Information Theory Entropy index of data in input
    Entropy rate for time series: k is the length of the time
    slice or interval we are considering
    """
    if not isinstance(k, int):
        raise ValueError("k [%s] has to be an integer" % k)
    if k < 1:
        raise ValueError("k [%s] has to be positive" % k)
    if k > 1:
        return entropy([L[i:i + k] for i in range(len(L) - k + 1)]) / k
    return sum(map(singleinf, probability(L)))


def mutualinfo(L1, L2, k=False):
    """
    Information Theory Mutual Information index
    between two data series
    """
    if not isinstance(k, int):
        raise ValueError("k [%s] has to be an integer" % k)
    if k < 1:
        raise ValueError("k [%s] has to be positive" % k)
    if k > 1:
        return multinfo(map(None, *(L1, L2)), k)
    return entropy(L1) + entropy(L2) - entropy(map(None, *(L1, L2)))


def multinfo(L, k=False):
    """
    Calculate Multi Information index.
    In input we have matrix L, with shape [[var1, ..., var n]
    in the i-th sample, ..., with n-variate data
    """
    if not isinstance(k, int):
        raise ValueError("k [%s] has to be an integer" % k)
    if k < 1:
        raise ValueError("k [%s] has to be positive" % k)
    if k > 1:
        return sum(map(entropy, map(None, *L),
                       itertools.repeat(k, len(L[0])))) - entropy(L, k)
    return sum(map(entropy, map(None, *L))) - entropy(L)
