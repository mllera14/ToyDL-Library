import numpy as np
from dataset.dataset import Dataset
from tools.rng_tools import get_numpy_rng


class KFold(object):
    def __init__(self, dataset, k=5, np_rng=None, shuffle=True):
        self._dataset = dataset
        self._k = k
        self._i = 0

        self._rng = get_numpy_rng(np_rng) if shuffle else None
        self._shuffle = shuffle

    def __iter__(self):
        indices = np.arange(self._dataset.size)

        x, y = self._dataset.data

        if self._shuffle:
            self._rng.shuffle(indices)
            x, y = x[indices], y[indices]

        return KFold(Dataset(x, y, self._dataset.problem_type), self._k, shuffle=False)

    def next(self):
        if self._i == self._k:
            raise StopIteration

        x, y = self._dataset.data
        fold_size = self._dataset.size / self._k

        v_set_x = x[self._i * fold_size: (self._i + 1) * fold_size]
        v_set_y = y[self._i * fold_size: (self._i + 1) * fold_size] if y is not None else None

        t_set_x = np.concatenate(
            [x[j * fold_size: (j + 1) * fold_size] for j in range(self._k) if j != self._i]
        )
        t_set_y = np.concatenate(
            [y[j * fold_size: (j + 1) * fold_size] for j in range(self._k) if j != self._i]
        ) if y is not None else None

        self._i += 1

        return Dataset(t_set_x, t_set_y, self._dataset.problem_type), \
            Dataset(v_set_x, v_set_y, self._dataset.problem_type)


def cross_validate(model, trainer, kfold):
    info = [trainer.train(model, t_set, v_set) for (t_set, v_set) in kfold]
    return np.mean([i.last_iter['validation_error'] for i in info])
