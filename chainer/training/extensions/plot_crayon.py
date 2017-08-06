import six


from chainer.training.extensions import plot_base


class PlotCrayon(plot_base.PlotBase):

    def __init__(
            self, crayon, experiment, y_keys, x_key='iteration',
            trigger=(1, 'epoch')):

        super(PlotCrayon, self).__init__(y_keys, x_key, trigger)
        self._crayon = crayon
        self._experiment = crayon.create_experiment(experiment)

    def plot(self, trainer, summary, x, ys):
        for key, y in six.iteritems(ys):
            if y is None:
                continue
            self._experiment.add_scalar_value(key, y, step=x)
