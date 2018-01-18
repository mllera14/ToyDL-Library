import pickle as pkl

import numpy as np
from PIL import Image
from theano.tensor.shared_randomstreams import RandomStreams

from dataset.dataset import Dataset
from benchmarks.mnist_loader import load_subset
from neural_net.autoencoder import dA, SdA, dAReconstructionError
from neural_net.mlp import MLP
from optimization.algorithm import RMSprop
from optimization.cost import CatXENT, ZeroOneLoss
from tools.plot import tile_raster_images
from training.extensions import EarlyStopper, ProgressTracker
from training.trainer import Trainer, GreedyLayerWiseTrainer
from wrapper.classifier import build_classifier, test_classifier


def test_autoencoder():
    classes = range(10)

    data_sets = Dataset(load_subset(
        training_count=6000,
        testing_count=1000,
        validation_percent=0.2,
        classes=classes
    )[0].inputs)

    rng = np.random.RandomState(1234)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    corruption_level = 0.3

    da = dA(
        n_features=28 * 28,
        code_size=500,
        np_rng=rng,
        theano_rng=theano_rng,
    )

    trainer = Trainer(
        algorithm=RMSprop(batch_size=20),
        cost_fn=dAReconstructionError(l1=0.0, l2=0.001, corruption_level=corruption_level, theano_rng=theano_rng),
        epochs=100
    )

    validation_error = trainer.train(da, data_sets)
    print('Best validation error on training was {0:.2%}'.format(validation_error))

    image = Image.fromarray(tile_raster_images(
        X=da.theta.get_value(borrow=True).T,
        img_shape=(28, 28),
        tile_shape=(10, 10),
        tile_spacing=(1, 1)
    ))
    image.save('filters_corruption_layer{0}.png'.format(corruption_level))


def test_stacked_autoencoder():
    classes = range(10)

    data_sets = load_subset(
        training_count=6000,
        testing_count=1000,
        validation_percent=0.2,
        classes=classes
    )

    classes = np.asarray([str(c) for c in classes])
    rng = np.random.RandomState(89677)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    sda = SdA(
        n_inputs=28 * 28,
        hidden_dims=[1000, 1000, 1000],
        np_rng=rng,
        theano_rng=theano_rng
    )

    trainer = GreedyLayerWiseTrainer(
        algorithm=RMSprop(alpha=0.001, batch_size=10),
        cost_fn=[
            dAReconstructionError(l1=0.0, l2=0.0001, corruption_level=0.1, theano_rng=theano_rng),
            dAReconstructionError(l1=0.0, l2=0.0001, corruption_level=0.2, theano_rng=theano_rng),
            dAReconstructionError(l1=0.0, l2=0.0001, corruption_level=0.3, theano_rng=theano_rng)
        ],
        validation_cost_fn=dAReconstructionError(),
        extensions=[EarlyStopper(patience=30, patience_factor=2, improvement_threshold=0.995), ProgressTracker()],
        epochs=15

    )

    trainer.train(sda, data_sets[0].get_unsupervised_dataset(), data_sets[1].get_unsupervised_dataset())
    # print('Best validation error on training was {0:.2%}'.format(validation_error))

    mlp = MLP(
        n_features=28 * 28,
        output_units=10,
        hidden_layers=[1000, 1000, 1000],
        params=sda.feed_forward_params,
        rng=rng
    )

    trainer = Trainer(
        algorithm=RMSprop(alpha=0.01, batch_size=20),
        cost_fn=CatXENT(l1=0.0, l2=0.0001),
        validation_cost_fn=ZeroOneLoss(),
        epochs=36
    )
    trainer.train(mlp, data_sets[0], validation_set=data_sets[1])

    with open('F:\\Miltonjr\\UH\\5th Year\\AI\\My Stuff\\MachineLearning\\benchmarks\\results\\autoencoder\\'
              'Predictions\\1000x3.txt', 'wb') as f:
        pkl.dump(mlp, f)

    classifier = build_classifier(mlp.propagate, classes)
    report = test_classifier(classifier, data_sets[2], classes, False)
    print(report)


if __name__ == '__main__':
    # test_autoencoder()
    test_stacked_autoencoder()
