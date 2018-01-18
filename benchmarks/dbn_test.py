import pickle

import numpy as np
import theano
import theano.tensor as T
from PIL import Image
from theano.tensor.shared_randomstreams import RandomStreams

from benchmarks.mnist_loader import load_subset
from neural_net.mlp import MLP
from neural_net.rbm import DBN
from optimization.algorithm import RMSprop
from optimization.cost import NLL, CDK, ZeroOneLoss
from tools.plot import tile_raster_images
from training.extensions import EarlyStopper, ProgressTracker
from training.trainer import Trainer, GreedyLayerWiseTrainer
from wrapper.classifier import build_classifier, test_classifier


def pre_train_dbn(dbn, data_sets, pt_epochs=20, pt_batch_size=20, pt_lr=0.01, gibbs_steps=1, save=False):
    print('Starting pretraining......\nEpochs: {0}\nBatch size: {1}\nLearning rate: {2}\nk: {3}'.format(
        pt_epochs, pt_batch_size, pt_lr, gibbs_steps)
    )

    rbmTM = GreedyLayerWiseTrainer(
        algorithm=RMSprop(alpha=pt_lr, batch_size=pt_batch_size),
        cost_fn=CDK(l2=0.0001, k=gibbs_steps, persistent=True),
        extensions=[ProgressTracker()],
        epochs=pt_epochs
    )

    rbmTM.train(model, data_sets[0].get_unsupervised_dataset())

    if save:
        print('Saving trained model.....')
        with open('dbn.txt', 'w') as f:
            pickle.dump(dbn, f)


def sample_dbn(dbn, data_sets, n_samples, sampling_interval=1000, n_plots=10):
    rng = np.random.RandomState(1234)

    print('Sampling.....')
    print('plots: {0}\nImages per plot: {1}\nPlotting interval: {2}'.format(n_plots, n_samples, sampling_interval))
    indices = rng.choice(data_sets[2].inputs.shape[0], size=n_samples, replace=True)

    x = T.dmatrix('input')
    output = x
    for rbm in dbn.layers:
        output, _ = rbm.sample_hgv(output)

    # Select 'samples' number of samples randomly from test set
    init_samples = theano.function(inputs=[x], outputs=output)(data_sets[2].inputs[indices, :])

    output, updates = dbn.generate_sample(k=sampling_interval, n_samples=n_samples, init_values=init_samples)
    sampling_function = theano.function(
        inputs=[],
        outputs=output,
        updates=updates
    )

    samples = []
    for i in range(n_plots):
        samples.append(sampling_function())

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


def supervised_fine_tune_dbn(dbn, classes, data_sets, st_epochs=100, st_batch_size=20, st_lr=0.01, save=True,):
    rng = np.random.RandomState(1802)

    print('Straing supervised training.......')
    print('Epochs: {0}\nBatch size: {1}\nLearning rate: {2}'.format(st_epochs, st_batch_size, st_lr))

    hidden_layer_sizes = [rbm.theta.get_value(borrow=True).shape[1] for rbm in dbn.layers]

    mlp = MLP(
        n_features=28 * 28,
        output_units=len(classes),
        hidden_layers=hidden_layer_sizes,
        params=dbn.feed_forward_params,
        activation_function=T.nnet.sigmoid,
        rng=rng
    )

    trainer = Trainer(
        algorithm=RMSprop(batch_size=st_batch_size),
        cost_fn=NLL(l2=0.001),
        validation_cost_fn=ZeroOneLoss(),
        extensions=[EarlyStopper(patience=10000, patience_factor=2, improvement_threshold=0.995), ProgressTracker()],
        epochs=st_epochs
    )

    validation_error = trainer.train(mlp, data_sets[0], data_sets[1])
    print('Supervised training ended\nBest validation error on training was {0:.2%}'.format(validation_error))

    classes = np.asarray([str(c) for c in classes])
    classifier = build_classifier(mlp.propagate, classes)

    report = test_classifier(classifier, data_sets[2], classes, False)
    print(report)

    if save:
        print('Saving trained MLP classifier initialized with model......')
        with open('dbn_classifier.txt', 'w') as f:
            pickle.dump(mlp, f)

    return mlp


def test_mlp_classifier(mlp, classes, test_set):
    classes = np.asarray([str(c) for c in classes])
    classifier = build_classifier(mlp.activation, classes)

    report = test_classifier(classifier, test_set, classes, False)
    print(report)


def plot_dbn_filters(dbn):
    for i, rbm in enumerate(dbn.layers):
        w = rbm.theta.get_value(borrow=True).T
        sqrt_dim = np.int(np.sqrt(w.shape[1]))
        img_shape = sqrt_dim, sqrt_dim
        image = Image.fromarray(tile_raster_images(
            X=w,
            img_shape=img_shape,
            tile_shape=(10, 10),
            tile_spacing=(1, 1)
        ))
        image.save('filters_layer{0}_[{1},{2}].png'.format(i, rbm.n_visible, rbm.n_hidden))


classes = range(10)

data_sets = load_subset(
    training_count=6000,
    testing_count=1000,
    validation_percent=0.2,
    classes=classes
)

np_rng = np.random.RandomState(1234)
theano_rng = RandomStreams(np_rng.randint(2 ** 30))

# dbn_path = instances + 'dbn[500,500,250].txt'
# with open(dbn_path, 'r') as f:
#     dbn = pickle.load(f)

model = DBN(
    n_inputs=28 * 28,
    hidden_info=[500, 500, 500],
    np_rng=np_rng,
    theano_rng=theano_rng
)

# path = instances + 'dbn[500,500,500].txt'
#
# with open(path) as f:
#     dbn = pickle.load(f)

pre_train_dbn(model, data_sets, pt_epochs=20, save=False)
sample_dbn(model, data_sets, 20)
supervised_fine_tune_dbn(model, classes, data_sets, st_epochs=20, save=False)
