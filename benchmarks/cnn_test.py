import numpy as np

from benchmarks.mnist_loader import load_subset
from neural_net.cnn import LeNetCNN
from optimization.algorithm import RMSprop
from optimization.cost import NLL, ZeroOneLoss
from training.extensions import EarlyStopper, ProgressTracker
from training.trainer import Trainer
from wrapper.classifier import build_classifier, test_classifier

classes = range(10)
data_sets = load_subset(
    training_count=6000,
    testing_count=1000,
    validation_percent=0.2,
    classes=classes
)

classes = np.asarray([str(c) for c in classes])
rng = np.random.RandomState(1234)

lnet = LeNetCNN(image_shape=(1, 28, 28), n_kernels=[4, 6], kernel_shapes=[(5, 5), (5, 5)], pool_sizes=[(2, 2), (2, 2)],
                n_output_units=10, hidden_layers=[500], rng=rng)

trainer = Trainer(
    RMSprop(batch_size=20), NLL(l1=0.0, l2=0.0001), monitoring_cost_fn=ZeroOneLoss(),
    extensions=[EarlyStopper(patience=40, patience_factor=2, improvement_threshold=0.995), ProgressTracker()],
    epochs=20
)

validation_error = trainer.train(lnet, data_sets[0], data_sets[1])
print('Best validation error on training was {0}'.format(validation_error))

# lnet.set_batch_size(len(data_sets[2].inputs))
classifier = build_classifier(lnet.propagate, classes)

report = test_classifier(classifier, data_sets[2], classes, False)
print(report)
