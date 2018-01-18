from numpy.random import RandomState
from theano.tensor.shared_randomstreams import RandomStreams


def get_numpy_rng(seed=None):
    if isinstance(seed, RandomState):
        return seed
    elif isinstance(seed, int) or seed is None:
        return RandomState(seed)
    raise ValueError('Value of seed is not of type int, RandomState or None')


def get_theano_rng(seed=None):
    if isinstance(seed, RandomStreams):
        return seed
    elif isinstance(seed, int):
        return RandomStreams(seed)

    return RandomStreams(get_numpy_rng(seed).randint(2 ** 30))
