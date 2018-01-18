import numpy as np
import theano.tensor as T

from calculator_test.sum_data_loader import load_dataset
from neural_net.rnn import ERNN
from optimization.algorithm import RMSprop
from optimization.cost import MSE
from training.trainer import Trainer
from wrapper.calculator import build_calculator

rng = np.random.RandomState(123)

training_data, test_data = load_dataset(rng)

rnn = ERNN(n_inputs=2, n_hidden=3, n_out=1, output_fn=T.nnet.sigmoid, numpy_rng=rng)
trainer = Trainer(RMSprop(batch_size=30), MSE(), epochs=30)

trainer.train(rnn, training_data)

sum_fn = build_calculator(rnn)

right = 0.0
for (i, j), t in zip(test_data.inputs, test_data.targets):
    s = sum_fn(i, j)
    print('{0} + {1} = {2}'.format(i, j, s))
    if s == t:
        right += 1.0

print('{0:.2%}'.format(right / len(test_data.targets)))
