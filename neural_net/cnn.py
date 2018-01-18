import numpy as np
import theano
import theano.tensor as T
from neural_net.layer import ConvolutionPoolLayer
from neural_net.mlp import MLP


class LeNetCNN(object):
    """ LeNet Convolutional Neural Network Model """
    def __init__(self, image_shape, n_kernels, kernel_shapes, pool_sizes, hidden_layers, n_output_units, act_fn=T.tanh,
                 output_type='classification', rng=None):
        assert len(kernel_shapes) == len(n_kernels) == len(pool_sizes)

        if len(image_shape) == 2:
            image_shape = (1, image_shape[0], image_shape[1])

        if rng is None:
            rng = np.random.RandomState()

        input_maps, img_shp = image_shape[0], image_shape[1:]
        n_kernels = [input_maps] + n_kernels

        self.conv_pool_layers = []
        self.params = []
        self.img_shape, self.input_channels = img_shp, image_shape[0]

        fmap_shapes = [img_shp]
        for p_size, k_shape in zip(pool_sizes, kernel_shapes):
            last_shape = fmap_shapes[-1]
            # Convolution reduces input dimension by one less the filter dimension in the corresponding axis
            # Ej (28, 28) image with (5, 5) filters produces feature maps of size (28 - 5 + 1, 28 - 5 + 1) = (24, 24)
            # Pooling divides de dimensions by the pooling factor in that axis
            # Ej (24, 24) pooled by a (2, 2) window produces a (12, 12) output
            next_shape = (last_shape[0] - k_shape[0] + 1) // p_size[0], (last_shape[1] - k_shape[1] + 1) // p_size[1]
            fmap_shapes.append(next_shape)

        for i in range(len(n_kernels) - 1):
            layer = ConvolutionPoolLayer(
                n_filters=n_kernels[i + 1],
                n_input_fmaps=n_kernels[i],
                filter_shape=kernel_shapes[i],
                image_shape=fmap_shapes[i],
                pool_size=pool_sizes[i],
                rng=rng
            )
            self.conv_pool_layers.append(layer)
            self.params.extend(layer.params)

        self.mlp = MLP(
            n_features=n_kernels[-1] * np.product(fmap_shapes[-1]),
            output_units=n_output_units,
            hidden_layers=hidden_layers,
            activation_function=act_fn,
            output_layer=output_type,
            rng=rng
        )

        self.params.extend(self.mlp.params)

    def __call__(self, inputs):
        return self.propagate(inputs)

    @property
    def theta(self):
        return [layer.theta for layer in self.conv_pool_layers] + self.mlp.theta

    @property
    def bias(self):
        return [layer.bias for layer in self.conv_pool_layers] + self.mlp.bias

    @property
    def input_shape(self):
        if self.input_channels != 1:
            return np.product(self.input_channels, self.img_shape[0], self.img_shape[1]),
        return np.product(self.img_shape),

    def propagate(self, inputs):
        # if batch_size is not None:
        #     temp = self.batch_size
        #     self.set_batch_size(batch_size)

        x = inputs.reshape((inputs.shape[0], self.input_channels, self.img_shape[0], self.img_shape[1]))
        x = x.dimshuffle([0, 'x', 1, 2, 3])

        def prop(ex):
            for layer in self.conv_pool_layers:
                ex = layer.activation(ex)
            return ex

        convolved, _ = theano.scan(fn=prop, sequences=[x])
        convolved = convolved.dimshuffle([0, 2, 3, 4])

        # if batch_size is not None:
        #     # noinspection PyUnboundLocalVariable
        #     self.set_batch_size(temp)

        return self.mlp.propagate(convolved.flatten(2))

    # def set_batch_size(self, value):
    #     self.batch_size = value
    #     for layer in self.conv_pool_layers:
    #         layer.set_batch_size(value)
