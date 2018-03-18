# Copyright 2016    Zakaria Aldeneh  University of Michigan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from chaipy import io
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict, Iterable
from keras.callbacks import Callback
from keras import backend as K


class CSVLogger(Callback):
    """ This is a slightly modified version of the CSVLogger used in the
    Keras Library. It was modified to work with training pipelines where
    'epochs' is given with all other metrics in the logs. """
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs={}):
        if self.append:
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

    def on_epoch_end(self, epoch, logs={}):
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(lambda x: str(x), k)))
            else:
                return k

        if not self.writer:
            self.keys = logs.keys()
            self.writer = csv.DictWriter(self.csv_file, self.keys)
            self.writer.writeheader()

        row_dict = OrderedDict()
        row_dict.update((key, handle_value(logs[key][-1])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs={}):
        self.csv_file.close()


class RestoreWeightsDropLR(Callback):
    """ The callback does the following depending on the options provided:
        (1) monitors how a metric changes between epochs and updates lr
        (2) saves network's best weights after each epoch
        (3) restores network's weights after each epoch"""
    def __init__(self, key_monitor=None, mode='min', patience=5,
                 weights_path=None, curr_lr=None, k=1, restore=1, min_lr=0.00001):
        self.key_monitor = key_monitor
        self.patience = patience
        self.weights_path = weights_path
        self.curr_lr = curr_lr
        self.k = float(k)
        self.restore = restore
        self.min_lr = min_lr
        self.best_val = None

        if mode == 'min':
            self.monitor_op = np.less_equal
        elif mode == 'max':
            self.monitor_op = np.greater_equal

        super(RestoreWeightsDropLR, self).__init__()

    def on_train_begin(self, logs={}):
        self.wait = 0

    def on_epoch_end(self, epoch, logs={}):

        monitor_values = logs[self.key_monitor]

        if len(monitor_values) >= 2:
            io.log('{}: {} --> {}'.format(self.key_monitor, self.best_val,
                                          monitor_values[-1]))

        if len(monitor_values) < 2 or self.monitor_op(monitor_values[-1],
                                                      self.best_val):
            io.log('saving weights')
            self.model.save_weights(self.weights_path, overwrite=True)
            self.best_val = monitor_values[-1]
            io.log('best {}: {}'.format(self.key_monitor, self.best_val))
            self.wait = 0

        else:
            if self.wait >= self.patience or self.curr_lr < self.min_lr:
                self.model.stop_training = True

            self.wait += 1

            if self.restore > 0:
                io.log('loading weights from last epoch')
                self.model.load_weights(self.weights_path)

            if self.k > 1.:
                self.curr_lr /= self.k
                K.set_value(self.model.optimizer.lr, self.curr_lr)
                io.log('lr: {} --> {}'.format(self.curr_lr*self.k, self.curr_lr))


class PlotMetrics(Callback):
    """This callback is used to plot metrics vs. epoch at the end of each
    epoch in a pdf format to a given directory."""
    def __init__(self, metrics, path):
        self.metrics = metrics
        self.path = path

    def create_plot(self, path, x_values, y_values, x_label, y_label):
        plt.clf()
        plt.plot(x_values, y_values)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim([0, len(x_values)+1])
        plt.savefig(path + '.pdf')

    def on_epoch_end(self, epoch, logs={}):

        x_values = logs['epoch']
        for metric in self.metrics:
            y_values = logs[metric]
            plot_path = self.path + '/' + metric
            self.create_plot(plot_path, x_values, y_values, 'epoch', metric)


class PrintLogs(Callback):
    """This callback prints the logs to io.log"""
    def __init__(self, metrics):
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs={}):

        for metric in self.metrics:
            values = logs[metric]
            io.log('{}: {}'.format(metric, values[-1]))
