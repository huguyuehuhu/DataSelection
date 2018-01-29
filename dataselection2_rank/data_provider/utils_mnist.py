# encoding: utf-8

"""
@author: huguyuehuhu
@time: 17-9-26 下午3:24
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


class DataTuple:
    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic minsttesting.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self.prob_images = images
        self.prob_labels = labels
        self.prob_index = []
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.load_start = True

        # characters of data_set
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    # next batch

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    def next_batch_with_prob(self, batch_size, prob, begin_shuffle=True):
        """Return the next `batch_size` examples from this data set.
            load_shuffle only shuffle before the first batch(load_start =True)
        """
        import numpy as np
        start = self._index_in_epoch
        # Shuffle for the only imagesthe first epoch
        if self.load_start and begin_shuffle and start == 0:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
            # self.load_start = False
        # generate samples depended on probability prob
        if start == 0:
            if self.load_start:
                self.prob_index = np.arange(self._num_examples)
                self.prob_images = self._images[self.prob_index]
                self.prob_labels = self._labels[self.prob_index]
                self.load_start = False

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self.prob_images[start:self._num_examples]
            labels_rest_part = self.prob_labels[start:self._num_examples]
            index_rest_part = self.prob_index[start:self._num_examples]
            start = 0

            self.prob_index = np.random.choice(self._num_examples, size=self._num_examples, p=prob)
            self.prob_images = self._images[self.prob_index]
            self.prob_labels = self._labels[self.prob_index]

            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.prob_images[start:end]
            labels_new_part = self.prob_labels[start:end]
            index_new_part = self.prob_index[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0), numpy.concatenate(
                (index_rest_part, index_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.prob_images[start:end], self.prob_labels[start:end], self.prob_index[start:end]


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
      f: A file object that can be passed into a gzip reader.

    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
      ValueError: If the bytestream does not start with 2051.

    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
      f: A file object that can be passed into a gzip reader.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.

    Returns:
      labels: a 1D uint8 numpy array.

    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels


def read_data_sets(train_dir,
                   one_hot=True,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    with open(train_dir+TRAIN_IMAGES, 'rb') as f:
        train_images = extract_images(f)

    with open(train_dir+TRAIN_LABELS, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    with open(train_dir+TEST_IMAGES, 'rb') as f:
        test_images = extract_images(f)

    with open(train_dir+TEST_LABELS, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.',
            format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return DataTuple(train=train, validation=validation, test=test)


def load_mnist(**params):
    print('data_params=%s' % params)
    return read_data_sets(**params)


if __name__ == "__main__":
    mnist = load_mnist(train_dir='/data/gyhu/code/mnist/', dtype=dtypes.float32,
                       one_hot=True, reshape=True, validation_size=5000)
    import matplotlib.pyplot as plt
    import numpy as np

    # plt.figure()
    # d, l = mnist.train.next_batch(batch_size=10, shuffle=False)
    # print(np.shape(d), np.shape(l))
    # img = np.reshape(d,(10,28,28))
    # print(img[0].dtype)
    # for i in range(10):
    #     plt.subplot(1,10,i+1)
    #     plt.imshow(img[i], cmap='gray')
    # plt.show()
    #
    # d, l = mnist.validation.next_batch(batch_size=10, shuffle=True)
    # print(np.shape(d), np.shape(l))
    # img = np.reshape(d, (10, 28, 28))
    # print(img[0].dtype)
    # for i in range(10):
    #     plt.subplot(2, 10, i + 1)
    #     plt.imshow(img[i], cmap='gray')
    # plt.show()
    # d, l = mnist.test.next_batch(batch_size=10, shuffle=True)
    # print(np.shape(d), np.shape(l))
    # img = np.reshape(d, (10, 28, 28))
    # print(img[0].dtype)
    # for i in range(10):
    #     plt.subplot(3, 10, i + 1)
    #     plt.imshow(img[i], cmap='gray')
    # plt.show()

    plt.figure()
    prob = np.zeros(mnist.train.num_examples)
    prob[0:3] = [0.4, 0.4, 0.2]
    batch_size = 20
    d, l, index = mnist.train.next_batch_with_prob(batch_size=batch_size, prob=prob, begin_shuffle=True)
    print(np.shape(d), np.shape(l))
    print(index)
    img = np.reshape(d, (batch_size, 28, 28))
    print(img[0].dtype)
    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.imshow(img[i], cmap='gray')
    plt.show()