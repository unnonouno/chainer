import operator

import chainer
import chainer.functions as F
import chainer.links as L
import numpy


class BatchSequence(object):

    def __init__(self, data, lengths, batch_dim, length_dim):
        assert lengths.ndim == 1
        assert len(lengths) == data.shape[batch_dim]
        assert (lengths <= data.shape[length_dim]).all()

        self._data = data
        self._lengths = lengths
        self._batch_dim = batch_dim
        self._length_dim = length_dim

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    def _apply_binop(self, f, rhs):
        data = self._data
        print(data.shape)
        if isinstance(rhs, chainer.Variable):
            data, rhs = F.broadcast(data, rhs)
            rhs = BatchSequence(
                rhs,
                numpy.full(self._data.shape[self._batch_dim],
                           self._data.shape[self._length_dim], 'i'),
                self._batch_dim, self._length_dim)
        data, r_data = F.broadcast(data, rhs._data)

        assert isinstance(rhs, BatchSequence)
        assert data.shape == r_data.shape, (data.shape, r_data.shape)
        assert self._batch_dim == rhs._batch_dim
        assert self._length_dim == rhs._length_dim
        data = f(data, r_data)
        print(self._lengths)
        print(rhs._lengths)
        lengths = numpy.minimum(self._lengths, rhs._lengths)
        return BatchSequence(
            data, lengths, self._batch_dim, self._length_dim)

    def __mul__(self, rhs):
        return self._apply_binop(operator.mul, rhs)

    def apply_reduce(self, f, axis):
        if axis == self._batch_dim:
            raise
        elif axis == self._length_dim:
            index = [slice(None, None)] * self._data.ndim
            result = []
            for i, l in enumerate(self._lengths):
                index[self._batch_dim] = i
                index[self._length_dim] = slice(0, l)
                y = f(F.get_item(self._data, index), axis=axis, keepdims=True)
                result.append(y)
            return F.concat(result, axis=axis)
        else:
            data = f(self._data, axis=axis)
            lengths = self._lengths.copy()
            return BatchSequence(
                data, lengths, self._batch_dim, self._length_dim)

    def sum(self, axis):
        return self.apply_reduce(F.sum, axis)

    def __getitem__(self, index):
        return BatchSequence(
            self._data[index], self._lengths, self._batch_dim, self._length_dim)

    def each_sequence(self):
        index = [slice(None, None)] * self._data.ndim
        for i, l in enumerate(self._lengths):
            index[self._batch_dim] = i
            index[self._length_dim] = slice(0, l)
            yield F.get_item(self._data, index)

    def raw():
        pass

    def transpose(self):
        lengths = numpy.array([
            (self._lengths > i).sum()
            for i in range(self._data.shape[self._length_dim])], 'i')
        return BatchSequence(
            self._data, lengths, self._length_dim, self._batch_dim)
        


class Decoder(chainer.Chain):

    def __init__(self):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.lstm = L.StatelessLSTM(5)
            self.W = L.Linear(10, 5)

    def __call__(self, hxs, hy, eys):
        cy = None
        for ey in eys.each_sequence():
            #assert ey.shape == (4, 5), ey.shape
            batch = 4
            cy, h = self.lstm(cy, hy, ey)
            assert h.shape == (batch, 5)
            scores = (hxs * h).sum(axis=2)
            assert scores.shape == (3, batch)
            #alpha = F.softmax(scores, axis=0)
            alpha = scores
            c = (alpha[:, :, None] * hxs).transpose().sum(axis=0)
            assert c.shape == (batch, 5), c.shape
            hy = F.tanh(self.W(F.concat([c, hy])))

if __name__ == '__main__':
    model = Decoder()

    hxs = numpy.zeros((3, 4, 5), dtype='f')
    hxs = BatchSequence(hxs, numpy.array([4, 4, 2], 'i'), 0, 1)
    hy = numpy.zeros((4, 5), dtype='f')
    eys = numpy.zeros((3, 4, 5), dtype='f')
    eys = BatchSequence(eys, numpy.array([4,3,1], 'i'), 0, 1)
    model(hxs, hy, eys)
