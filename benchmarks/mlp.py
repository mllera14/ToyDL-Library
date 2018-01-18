import numpy as np

from benchmarks.mnist_loader import load_subset
from neural_net.mlp import MLP
from optimization.algorithm import NAG
from optimization.cost import NLL, ZeroOneLoss
from training.extensions import EarlyStopper, ProgressTracker
from training.trainer import Trainer
from wrapper.classifier import build_classifier, test_classifier


def test_mlp():
    classes = range(10)
    data_sets = load_subset(
        training_count=6000,
        testing_count=1000,
        validation_percent=0.2,
        classes=classes
    )

    classes = np.asarray([str(c) for c in classes])
    rng = np.random.RandomState(1234)

    mlp = MLP(
        n_features=28 * 28,
        output_units=10,
        hidden_layers=[1000, 1000],
        rng=rng
    )

    trainer = Trainer(
        # RMSprop(alpha=0.01, rho=0.9, batch_size=20),
        NAG(alpha=0.01, batch_size=20),
        NLL(l1=0.0, l2=0.0001),
        monitoring_cost_fn=ZeroOneLoss(),
        extensions=[EarlyStopper(patience=40, patience_factor=2, improvement_threshold=0.995), ProgressTracker()],
        epochs=5
    )

    training_history = trainer.train(mlp, data_sets[0], data_sets[1])

    print('Best validation error on training was {0}'.format(min(training_history.validation_errors())))

    classifier = build_classifier(mlp.propagate, classes)
    report = test_classifier(classifier, data_sets[2], classes, False)
    print(report)


if __name__ == '__main__':
    test_mlp()
