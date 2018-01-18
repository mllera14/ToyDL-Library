import numpy as np
import theano
import theano.tensor as T
from PIL import Image
from theano.tensor.shared_randomstreams import RandomStreams

from benchmarks.mnist_loader import load_subset
from neural_net.rbm import RBM
from optimization.algorithm import RMSprop
from optimization.cost import CDK
from tools.plot import tile_raster_images
from training.trainer import Trainer


def test_rbm():
    classes = range(10)

    data_sets = load_subset(
        training_count=6000,
        testing_count=2,
        validation_percent=0.2,
        classes=classes
    )

    rng = np.random.RandomState(1234)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    k, alpha, batch_size = 15, 0.0001, 20
    method = 'pcd-{0}'.format(k)

    rbm = RBM(
        n_visible=784,
        n_hidden=500,
        numpy_rng=rng,
        theano_rng=theano_rng,
    )

    print('Training.....\nParameters are:\n-optimization rate: {0}\n-Number of Gibbs steps: {1}'.format(alpha, k))

    trainer = Trainer(
        algorithm=RMSprop(batch_size=20),
        cost_fn=CDK(l1=0.0, l2=0.0, k=15, persistent=True),
        epochs=1
    )
    trainer.train(rbm, training_set=data_sets[0])

    w = rbm.W.get_value(borrow=True).T
    sqrt_dim = np.int(np.sqrt(w.shape[1]))
    img_shape = sqrt_dim, sqrt_dim
    image = Image.fromarray(tile_raster_images(
        X=w,
        img_shape=img_shape,
        tile_shape=(10, 10),
        tile_spacing=(1, 1)
    ))
    image.save('filters_rbm_{0}_alpha_{1}.png'.format(method, alpha))

    # Sampling from learned RBM
    print('Sampling.....')

    testing_examples = data_sets[2].inputs
    sample_size = testing_examples.shape[0]
    samples_to_plot = 10
    steps_between_samples = 1000

    print('Parameters are:\n-Number of sample chains: {0}\n-Steps between each sample plot: {1}'
          .format(sample_size, steps_between_samples))

    inputs = T.dmatrix('input')

    sample_hgv = theano.function(
        inputs=[inputs],
        outputs=rbm.sample_hgv(inputs)
    )

    init_samples, _ = sample_hgv(testing_examples)

    chain_state = theano.shared(
        value=init_samples,
        name='chain_state',
    )

    (v_samples, pre_act), updates = theano.scan(
        fn=rbm.gibbs_step_given_h,
        outputs_info=[chain_state, None],
        n_steps=steps_between_samples
    )
    chain_end = v_samples[-1]

    updates[chain_state] = chain_end

    sampling_step = theano.function(
        inputs=[],
        outputs=rbm.prop_down(chain_end)[0],
        updates=updates,
    )

    samples = []
    for i in range(samples_to_plot):
        samples.append(sampling_step())

    image_data = np.zeros(
        shape=(29 * 10 + 1, 29 * 20 - 1),
        dtype='uint8'
    )

    for i, s in enumerate(samples):
        image_data[29 * i: 29 * i + 28, :] = tile_raster_images(
            X=s,
            img_shape=(28, 28),
            tile_shape=(1, 20),
            tile_spacing=(1, 1)
        )

    image = Image.fromarray(image_data)
    image.save('samples.png')


if __name__ == '__main__':
    test_rbm()
