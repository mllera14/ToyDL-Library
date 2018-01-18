import theano
import theano.tensor as T
import numpy as np
from functools import reduce


def build_classifier(prob_fn, classes):
    x = T.dmatrix('classifier_input')
    propagate = theano.function([x], prob_fn(x))

    def classifier(inputs, mode='Number'):
        p_ys = propagate(inputs)
        i = np.argmax(p_ys, axis=1)

        if mode == 'Number':
            return i
        if mode == 'Name':
            return classes[i]
        if mode == 'Both':
            return i, classes[i]
        if mode == 'Testing':
            return i, classes[i], p_ys

    return classifier


def test_classifier(classifier, testing_examples, classes, verbose=False):
    test_x, test_y = testing_examples.inputs, testing_examples.targets

    predicted = classifier(test_x, 'Testing')
    right = reduce(lambda acc, val: acc + 1 if val else acc, np.equal(predicted[0], test_y), 0)
    example_list = []

    if verbose:
        predicted_classes = predicted[1]
        for i in range(len(test_y)):
            example_list.append((i, predicted_classes[i], classes[test_y[i]]))

    return Output(right, len(test_y), example_list)


class Output(object):
    def __init__(self, right, total, examples):
        self._examples = examples
        self._right = right
        self._total = total

    def __str__(self):
        example_list = []

        for e in self._examples:
            example_list.append("----------Test example: {0}----------\n"
                                "  Expected {1}, Got {2}\n".format(e[0], e[1], e[2]))

        example_list.append("----------Testing finished----------\n"
                            "{0}/{1} test examples correct\n"
                            "Precision: {2:.2%}".format(self._right, self._total, self.success_rate))

        return reduce(lambda x, r: x + r, example_list, "")

    @property
    def success_rate(self):
        return float(self._right) / self._total

    def dump(self, path):
        with open(path, 'w') as log:
            log.write(str(self))
