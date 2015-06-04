import types
import numpy
from chainer import cuda, Function, FunctionSet
import chainer.functions as F

def _get_ksize(param):
    if param.kernel_h > 0:
        return (param.kernel_h, param.kernel_w)
    else:
        return param.kernel_size

def _get_stride(param):
    if param.stride_h > 0:
        return (param.stride_h, param.stride_w)
    else:
        return param.stride

def _get_pad(param):
    if param.pad_h > 0:
        return (param.pad_h, param.pad_w)
    else:
        return param.pad
    

class CaffeFunction(Function):
    """Function using Caffe's model file."""

    def __init__(self, model_path):
        from caffe.proto import caffe_pb2

        net = caffe_pb2.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())

        if not net.layer:
            raise RuntimeError('Caffe model in old format. Upgrade it by upgrade_net_proto_binary.bin')

        self.fs        = FunctionSet()
        self.forwards  = {}
        self.split_map = {}
        self.layers    = []

        for layer in net.layer:
            typ      = layer.type
            methname = '_process_{}'.format(typ)
            meth     = getattr(self, methname, None)
            if meth:
                meth(layer)

    def __call__(self, inputs, outputs, disable=[], train=True):
        self.train = train
        variables = dict(inputs)
        for func_name, bottom, top in self.layers:
            print func_name, variables
            if (func_name in disable or
                any(blob not in variables for blob in bottom)):

                continue

            func = self.forwards[func_name]
            input_vars  = tuple(variables[blob] for blob in bottom)
            # TODO(beam2d): Support test mode for some functions (e.g. Dropout)
            output_vars = func(*input_vars)
            for var, name in zip(output_vars, top):
                variables[name] = var

        return tuple(variables[blob] for blob in outputs)

    def to_gpu(self, device=None):
        self.fs.to_gpu(device)
        return self

    def to_cpu(self):
        self.fs.to_cpu()
        return self

    @property
    def parameters(self):
        return self.fs.parameters

    @parameters.setter
    def parameters(self, values):
        self.fs.parameters = values

    @property
    def gradients(self):
        return self.fs.gradients

    @parameters.setter
    def gradients(self, values):
        self.fs.gradients = values

    def _add_layer(self, layer):
        bottom = []
        for blob_name in layer.bottom:
            bottom.append(self.split_map.get(blob_name, blob_name))
        self.layers.append((layer.name, layer.bottom, layer.top))

    def _process_Concat(self, layer):
        param = layer.concat_param
        axis  = param.axis
        if axis == 1 and param.concat_dim != 1:
            axis = param.concat_dim

        self.forwards[layer.name] = lambda *xs: F.concat(xs, axis=axis)
        self._add_layer(layer)

    def _process_Convolution(self, layer):
        blobs = layer.blobs
        param = layer.convolution_param

        ksize  = _get_ksize(param)
        stride = _get_stride(param)
        pad    = _get_pad(param)

        nobias = not param.bias_term

        if param.group == 1:
            func = F.Convolution2D(blobs[0].channels, param.num_output,
                                   ksize, stride, pad, nobias=nobias)
            func.W.ravel()[:] = blobs[0].data
            if param.bias_term:
                func.b[:] = blobs[1].data
            setattr(self.fs, layer.name, func)
            self.forwards[layer.name] = func
        else:
            funcs = FunctionSet()
            for i in xrange(param.group):
                func = F.Convolution2D(
                    blobs[0].channels, param.num_output / param.group,
                    ksize, stride, pad, nobias=nobias)
                setattr(funcs, str(i), func)
                func.W.ravel()[:] = blobs[0].data[i*func.W.size : (i+1)*func.W.size]
                func.b[:] = blobs[1].data[i*func.b.size : (i+1)*func.b.size]
            setattr(self.fs, layer.name, funcs)
            # TODO(beam2d): Implement forward function
            raise RuntimeError('grouped Convolution is not supported')

        self._add_layer(layer)

    def _process_Dropout(self, layer):
        param = layer.dropout_param

        self.forwards[layer.name] = lambda x: F.dropout(
            x, ratio=param.dropout_ratio, train=self.train)
        self._add_layer(layer)

    def _process_InnerProduct(self, layer):
        param = layer.inner_product_param
        if param.axis != 1:
            raise RuntimeError('Non-default axis in InnerProduct is not supported')

        blobs = layer.blobs
        func = F.Linear(blobs[0].width, blobs[0].height, nobias=not param.bias_term)
        func.W.ravel()[:] = blobs[0].data
        if param.bias_term:
            func.b[:] = blobs[1].data

        setattr(self.fs, layer.name, func)
        self.forwards[layer.name] = func
        self._add_layer(layer)

    def _process_LRN(self, layer):
        param = layer.lrn_param
        if param.norm_region != param.ACROSS_CHANNELS:
            raise RuntimeError('Within-channel LRN is not supported')

        self.forwards[layer.name] = lambda x: F.local_response_normalization(
            x, n=param.local_size, k=param.k, alpha=param.alpha, beta=param.beta)
        self._add_layer(layer)

    def _process_Pooling(self, layer):
        param = layer.pooling_param
        ksize  = _get_ksize(param)
        stride = _get_stride(param)
        pad    = _get_pad(param)

        if param.pool == param.MAX:
            fw = lambda x: F.max_pooling_2d(x, ksize, stride=stride, pad=pad)
        elif param.pool == param.AVE:
            fw = lambda x: F.average_pooling_2d(x, ksize, stride=stride, pad=pad)
        else:
            raise RuntimeError('Stochastic pooling is not supported')

        self.forwards[layer.name] = fw
        self._add_layer(layer)

    def _process_ReLU(self, layer):
        slope = layer.relu_param.negative_slope
        if slope != 0:
            fw = lambda x: F.leaky_relu(x, slope=slope)
        else:
            fw = F.relu

        self.forwards[layer.name] = fw
        self._add_layer(layer)

    def _process_SoftmaxWithLoss(self, layer):
        if layer.softmax_param.axis != 1:
            raise RuntimeError('Softmax along non-channel axis is not supported')

        self.forwards[layer.name] = F.softmax_cross_entropy

    def _process_Split(self, layer):
        for top in layer.top:
            self.split_map[top] = layer.bottom
