import theano
import theano.tensor as T
import numpy as np


def to_binary(i):
    return map(int, list(reversed(bin(i)[2:])))


def padded_bin(i):
    return to_binary(i) + np.asarray([0, 0])


def build_calculator(rnn):
    inputs = T.tensor3('inputs')

    rnn_output = T.cast(rnn(inputs)[0] >= 0.5, 'int32')

    result, _ = theano.scan(
        lambda bit, value, prior: T.cast(2 ** bit * value[0] + prior, 'int32'),
        sequences=[T.arange(rnn_output.shape[0]), rnn_output],
        outputs_info=T.cast(0, 'int32')
    )

    rnn_sum = theano.function(
        inputs=[inputs],
        outputs=result[-1],
        name='sum_rnn',
        allow_input_downcast=True
    )

    def sum_fn(a, b):
        a = padded_bin(a)
        b = padded_bin(b)

        if len(a) < len(b):
            a = a + [0] * (len(b) - len(a))

        if len(a) > len(b):
            b = b + [0] * (len(a) - len(b))

        params = np.transpose([a, b])
        params = params.reshape((1, params.shape[0], params.shape[1]))

        return rnn_sum(params)

    return sum_fn
