import theano.tensor as T
from neural_net.layer import DenseLayer, SoftmaxLayer, LinearLayer, LogisticLayer
from tools.rng_tools import get_numpy_rng


class MLP(object):
    @property
    def theta(self):
        return [layer.theta for layer in self.layers]

    @property
    def bias(self):
        return [layer.bias for layer in self.layers]

    @property
    def input_shape(self):
        return self.params[0].get_value(borrow=True).shape[:1]

    def __init__(self, n_features, output_units, hidden_layers, params=None, output_layer='classification',
                 activation_function=T.tanh, rng=None):
        rng = get_numpy_rng(rng)

        dims = [n_features] + hidden_layers
        self.layers = []
        w, b = None, None

        for i in range(1, len(dims)):
            if params is not None:
                w, b = params[i - 1]
            hidden_layer = DenseLayer(
                n_features=dims[i - 1],
                n_units=dims[i],
                rng=rng,
                w=w, b=b,
                activation_function=activation_function
            )
            self.layers.append(hidden_layer)

        if output_layer == 'linear':
            output_layer = LinearLayer
        elif output_layer == 'logistic':
            output_layer = LogisticLayer
        elif output_layer == 'classification':
            output_layer = SoftmaxLayer

        output_layer = output_layer(
            n_units=output_units,
            n_features=dims[-1]
        )

        self.layers.append(output_layer)

        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def __call__(self, inputs):
        return self.propagate(inputs)

    def propagate(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.activation(x)
        return x
