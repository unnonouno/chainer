import collections

from chainer import reporter
from chainer.training import extension
import chainer.training.trigger as trigger_module


class PlotBase(extension.Extension):

    def __init__(self, y_keys, x_key='iteration', trigger=(1, 'epoch')):
        self._x_key = x_key
        if isinstance(y_keys, str):
            y_keys = (y_keys,)

        self._y_keys = y_keys
        self._trigger = trigger_module.get_trigger(trigger)
        self._init_summary()

    def plot(self, trainer, summary, x, ys):
        raise NotImplemented

    def __call__(self, trainer):
        keys = self._y_keys
        observation = trainer.observation
        summary = self._summary

        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})

        if self._trigger(trainer):
            stats = self._summary.compute_mean()
            stats_cpu = collections.OrderedDict()
            for key in keys:
                # copy to CPU
                stats_cpu[key] = float(stats[key]) if key in stats else None

            if self._x_key == 'epoch':
                x = trainer.updater.epoch
            elif self._x_key == 'iteration':
                x = trainer.updater.iteration
            elif self._x_key in stats:
                x = float(stats[self._x_key])
            else:
                raise KeyError()

            self.plot(trainer, summary, x, stats_cpu)

            self._init_summary()

    def _init_summary(self):
        self._summary = reporter.DictSummary()
