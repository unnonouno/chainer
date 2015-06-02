import types
import numpy
from chainer import cuda, Function, FunctionSet
import chainer.functions as F

class CaffeFunction(Function):
    """Function using Caffe's model file."""

    def __init__(self, model_path):
        from caffe.proto import caffe_pb2

        net = caffe_pb2.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())

        if not net.layer:
            raise RuntimeError('Caffe model in old format. Upgrade it by upgrade_net_proto_binary.bin')

        self.fs = FunctionSet()
        self.split_map = {}
        self.layers    = []

        for layer in net.layer:
            typ      = layer.type
            methname = '_process_{}'.format(typ)
            meth     = getattr(self, methname, None)
            if meth:
                meth(layer)

    def __call__(self, inputs, outputs):
        variables = dict(inputs)
        # TODO(beam2d): Stop computation if all the outputs are computed.
        for func_name, bottom, top in self.layers:
            if any(blob not in variables for blob in bottom):
                continue

            func = getattr(self.fs, func_name)
            if isinstance(func, FunctionSet):  # grouped Convolution2D
                # TODO(beam2d): Implement split function along the channel dimension
                raise NotImplementedError()
                continue

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

    def _process_Convolution(self, layer):
        blobs = layer.blobs
        param = layer.convolution_param

        if param.kernel_h > 0:
            ksize = (param.kernel_h, param.kernel_w)
        else:
            ksize = param.kernel_size

        if param.stride_h > 0:
            stride = (param.stride_h, param.stride_w)
        else:
            stride = param.stride

        if param.pad_h > 0:
            pad = (param.pad_h, param.pad_w)
        else:
            pad = param.pad

        nobias = not param.bias_term

        if param.group == 1:
            func = F.Convolution2D(blobs[0].channels, param.num_output,
                                   ksize, stride, pad, nobias=nobias)
            func.W.ravel()[:] = blobs[0].data
            func.b[:] = blobs[1].data
            setattr(self.fs, layer.name, func)
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

        self._add_layer(layer)

    def _process_Split(self, layer):
        for top in layer.top:
            self.split_map[top] = layer.bottom
