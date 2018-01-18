import os
import struct
import numpy as np
from array import array
from dataset.dataset import Dataset


class MNIST(object):
    def __init__(self, train_path='.', test_path='.'):
        self.training_path = train_path
        self.test_path = test_path

        self.test_img_fname = 't10k-images.idx3-ubyte'
        self.test_lbl_fname = 't10k-labels.idx1-ubyte'

        self.train_img_fname = 'train-images.idx3-ubyte'
        self.train_lbl_fname = 'train-labels.idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self, cant=10000):
        ims, labels = self.load(os.path.join(self.test_path, self.test_img_fname),
                                os.path.join(self.test_path, self.test_lbl_fname))

        images = [self.normalize(x) for i, x in enumerate(ims) if i < cant]
        lab = [x for i, x in enumerate(labels) if i < cant]

        self.test_images = images
        self.test_labels = lab

        return self.test_images, self.test_labels

    def load_training(self, cant=60000):
        ims, labels = self.load(os.path.join(self.training_path, self.train_img_fname),
                                os.path.join(self.training_path, self.train_lbl_fname))

        images = [self.normalize(x) for i, x in enumerate(ims) if i < cant]
        lab = [x for i, x in enumerate(labels) if i < cant]

        self.train_images = images
        self.train_labels = lab

        return self.train_images, self.train_labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got %d' % magic)

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got %d' % magic)

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols: (i + 1) * rows * cols]

        return images, labels

    def test(self):
        test_img, test_label = self.load_testing()
        train_img, train_label = self.load_training()
        assert len(test_img) == len(test_label)
        assert len(test_img) == 10000
        assert len(train_img) == len(train_label)
        assert len(train_img) == 60000
        print('Showing num:%d' % train_label[0])
        print(self.display(train_img[0]))
        return True

    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render

    @staticmethod
    def normalize(xi):
        return [(x * 1.0 / 255.0) for x in xi]


def load_data(training_count=60000, testing_count=10000):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    mn = MNIST('../Datasets/MNIST/training',
               '../Datasets/MNIST/test')

    training_data = mn.load_training(training_count)
    validation_data, test_data = mn.load_testing(testing_count)

    return training_data, validation_data, test_data


def load_subset(training_count=6000, testing_count=1000, validation_percent=0.2, classes=None):
    training, validation, testing = load_data()

    classes = {c: 0 for c in range(10)} if classes is None else {c: 0 for c in classes}

    test_x, test_y = validation, testing
    test_x_subset, test_y_subset = [], []

    if testing_count >= 1000:
        test_x_subset, test_y_subset = test_x, test_y
    else:
        for i in range(10000):
            if test_y[i] in classes and classes[test_y[i]] < testing_count:
                classes[test_y[i]] += 1
                test_x_subset.append(test_x[i])
                test_y_subset.append(test_y[i])

    test_set = Dataset(np.asarray(test_x_subset), np.asarray(test_y_subset), problem_type='Classification')

    train_x, train_y = training
    train_subset = []

    classes = {c: 0 for c in range(10)} if classes is None else {c: 0 for c in classes}

    if testing_count >= 6000:
        train_subset = zip(train_x, train_y)
    else:
        for x, y in zip(train_x, train_y):
            if y in classes and classes[y] < training_count:
                classes[y] += 1
                train_subset.append((x, y))

    np.random.shuffle(train_subset)
    train_x_subset = [example[0] for example in train_subset]
    train_y_subset = [example[1] for example in train_subset]

    cut_index = int(len(train_subset) * (1 - validation_percent))

    train_set = Dataset(
        np.asarray(train_x_subset[:cut_index]), np.asarray(train_y_subset[:cut_index]), problem_type='Classification')
    validation_set = Dataset(
        np.asarray(train_x_subset[cut_index:]), np.asarray(train_y_subset[cut_index:]), 'Classification')

    return train_set, validation_set, test_set


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == "__main__":
    print('Testing')
    mnist = MNIST('F:/Miltonjr/UH/5th Year/AI/My Stuff/tools/MNIST/training',
                  'F:/Miltonjr/UH/5th Year/AI/My Stuff/tools/MNIST/test')
    if mnist.load_training():
        print('Passed')
