from chainer import cuda
from chainer import function


class SparseLinearFunction(function.Function):

    def __init__(self, n_in, n_out, i1, i2):
        self.n_in = n_in
        self.n_out = n_out
        self.i1 = i1
        self.i2 = i2

    def forward_gpu(self, inputs):
        x, W = inputs
        batch = x.shape[0]
        y = cuda.cupy.zeros((batch, self.n_out), dtype=x.dtype)
        cuda.elementwise(
            'raw T x, raw T w, raw int32 i1, raw int32 i2, int32 batch',
            'raw T y',
            '''
            int ind = i / batch;
            int b = i - ind * batch;
            int x_ind[] = {b, i1[ind]};
            int y_ind[] = {b, i2[ind]};
            atomicAdd(&y[y_ind], x[x_ind] * w[ind]);
            ''',
            'sparse_linear_fwd',
        )(x, W, self.i1, self.i2, batch, y, size=batch * self.i1.size)
        return y,

    def backward_gpu(self, inputs, grads):
        x, W = inputs
        gy, = grads
        gx = cuda.cupy.zeros_like(x)
        gw = cuda.cupy.zeros_like(W)
        batch = x.shape[0]
        cuda.elementwise(
            'raw T x, raw T w, raw T gy, raw S i1, raw S i2, int32 batch',
            'raw T gx, raw T gw',
            '''
            int ind = i / batch;
            int b = i - ind * batch;
            int x_ind[] = {b, i1[ind]};
            int y_ind[] = {b, i2[ind]};
            T g = gy[y_ind];
            atomicAdd(&gx[x_ind], g * w[ind]);
            atomicAdd(&gw[ind], g * x[x_ind]);
            ''',
            'sparse_linear_bwd',
        )(x, W, gy, self.i1, self.i2, batch, gx, gw, size=batch * self.i1.size)
        return gx, gw


def sparse_linear(x, W, i1, i2, n_in, n_out):
    return SparseLinearFunction(n_in, n_out, i1, i2)(x, W)
