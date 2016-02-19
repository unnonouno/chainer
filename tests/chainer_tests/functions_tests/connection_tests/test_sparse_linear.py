import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check


class TestSparseLinearFunction(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.W = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.i1 = numpy.array([0, 1, 0], numpy.int32)
        self.i2 = numpy.array([0, 2, 3], numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, (2, 4)).astype(numpy.float32)

    def check_forward(self, x_data, W_data, i1, i2):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        y = functions.sparse_linear(x, W, i1, i2, 3, 4)

        eW = numpy.zeros((3, 4)).astype(numpy.float32)
        for i, (i1, i2) in enumerate(zip(self.i1, self.i2)):
            eW[i1, i2] = self.W[i]

        ey = cuda.to_cpu(x_data).dot(eW)
        numpy.testing.assert_array_equal(cuda.to_cpu(y.data), ey)

    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.W),
            cuda.to_gpu(self.i1),
            cuda.to_gpu(self.i2))

    def check_backward(self, x_data, W_data, i1, i2, y_grad):
        args = (x_data, W_data)

        gradient_check.check_backward(
            functions.SparseLinearFunction(3, 4, i1, i2),
            args, y_grad, eps=1e-2)

    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.W),
            cuda.to_gpu(self.i1),
            cuda.to_gpu(self.i2),
            cuda.to_gpu(self.gy))
