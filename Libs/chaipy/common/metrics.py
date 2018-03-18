# Copyright 2015    Duc Le  University of Michigan
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

import numpy
from sklearn import metrics
from chaipy.keras import preds_to_classes
import keras.backend as K

import chaipy.common as common


#######################################
## Metrics compatible with Keras API ##
#######################################

def mse(y_true, y_pred):
    """ Compute MSE between two time series """
    return numpy.mean(numpy.square(y_pred - y_true))


def rmse(y_true, y_pred):
    """ Compute RMSE between two time series """
    return numpy.sqrt(numpy.mean(numpy.square(y_pred - y_true)))


def ccc(y_true, y_pred):
    """ Compute the concordance correlation coefficient (CCC) between two
    1D time series """
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    cov_xy = numpy.mean(y_pred*y_true, axis=1) - \
        (numpy.mean(y_pred, axis=1) * numpy.mean(y_true, axis=1))
    mean_x = numpy.mean(y_pred, axis=1)
    mean_y = numpy.mean(y_true, axis=1)
    var_x = numpy.var(y_pred, axis=1)
    var_y = numpy.var(y_true, axis=1)
    return (2*cov_xy/(var_x + var_y + numpy.square(mean_x - mean_y)))[0]


def ccc_err(y_true, y_pred):
    """ Error metric based on CCC
    """
    return 1.0 - ccc(y_true, y_pred)


def ccc_err_tensor(y_true, y_pred):
    """ Symbolic CCC error for two 1D time series. Can be used as objective.
    """
    cov_xy = K.mean(y_pred * y_true) - (K.mean(y_pred) * K.mean(y_true))
    mean_x = K.mean(y_pred)
    mean_y = K.mean(y_true)
    var_x = K.var(y_pred)
    var_y = K.var(y_true)
    return 1 - (2 * cov_xy / (var_x + var_y + K.square(mean_x - mean_y)))


def ccc_err_tensor_multi(y_true, y_pred):
    """ Symbolic CCC error for multiple 1D time series. This function returns
    mean CCC error for different time series pairs. Can be used as objective.
    """
    cccs, _ = K.theano.scan(
        fn=lambda yt, yp: ccc_err_tensor(yt, yp), outputs_info=None,
        sequences=[y_true, y_pred]
    )
    return cccs.mean()


############################################################
## Metrics that accumulate over a collection of instances ##
############################################################

class AbstractMetric(object):
    """ Base class for accumulating metrics.
    """
    def accum(self, y_true, y_pred):
        """ Accumulate stats for one pair of ground-truth and predicted labels.
        """
        raise NotImplementedError()

    def eval(self):
        """ Return the metric based on the accumulated stats thus far.
        """
        raise NotImplementedError()


class KerasMetric(AbstractMetric):
    """ Wrapper for Keras metric functions. Use the function for accumulating
    stats and return the average stats over all instances.
    """
    def __init__(self, fn):
        """
        :type fn: function
        :param fn: Keras-compatible metric function
        """
        self.fn = fn
        self.stats = []

    def accum(self, y_true, y_pred):
        self.stats.append(self.fn(y_true, y_pred))

    def eval(self):
        common.CHK_GT(len(self.stats), 0)
        return numpy.mean(self.stats)


class FrameError(AbstractMetric):
    """ Returns frame-level classification error (%).
    """
    def __init__(self):
        self.total = 0.0
        self.correct = 0.0

    def accum(self, y_true, y_pred):
        classes = preds_to_classes(y_pred)
        res = numpy.asarray(numpy.equal(y_true, classes), dtype=numpy.int)
        self.total += res.size
        self.correct += numpy.sum(res)

    def eval(self):
        common.CHK_GT(self.total, 0)
        acc = self.correct / self.total
        return 100 * (1.0 - acc)


class FrameUnweightedAverageError(AbstractMetric):
    """ Returns frame-level unweighted average classification error (%).
    """
    def __init__(self):
        self.total = {}
        self.correct = {}

    def accum(self, y_true, y_pred):
        classes = preds_to_classes(y_pred)
        common.CHK_EQ(len(y_true), len(classes))
        for corr, pred in zip(y_true, classes):
            if corr not in self.total:
                self.total[corr] = 0.0
                self.correct[corr] = 0.0
            self.total[corr] += 1
            self.correct[corr] += int(corr == pred)

    def eval(self):
        common.CHK_GT(len(self.total), 0)
        sum = 0
        for c in self.total:
            sum += self.correct[c] / self.total[c]
        uar = sum / len(self.total)
        return 100 * (1.0 - uar)


class UttError(FrameError):
    """ Returns utt-level classification error (%).
    """
    def accum(self, y_true, y_pred):
        corr = numpy.argmax(numpy.bincount(y_true))
        pred = preds_to_classes(numpy.sum(y_pred, axis=0))
        self.total += 1
        self.correct += int(corr == pred)


class UttUnweightedAverageError(FrameUnweightedAverageError):
    """ Returns utt-level unweighted average classification error (%).
    """
    def accum(self, y_true, y_pred):
        corr = numpy.argmax(numpy.bincount(y_true))
        pred = preds_to_classes(numpy.sum(y_pred, axis=0))
        if corr not in self.total:
            self.total[corr] = 0.0
            self.correct[corr] = 0.0
        self.total[corr] += 1
        self.correct[corr] += int(corr == pred)


class UttMeanSquaredError(AbstractMetric):
    """ Returns utterance-level mean squared error (%).
        """
    def __init__(self):
        self.error = 0.0
        self.total = 0.0

    def accum(self, y_true, y_pred):
        yhat_utt = numpy.mean(y_pred)
        y_utt = numpy.mean(y_true)
        self.error += numpy.square(y_utt - yhat_utt)
        self.total += 1

    def eval(self):
        common.CHK_GT(self.total, 0)
        return self.error / self.total


class UttMeanAbsoluteError(UttMeanSquaredError):
    """ Returns utterance-level mean absolute error (%).
        """
    def accum(self, y_true, y_pred):
        yhat_utt = numpy.mean(y_pred)
        y_utt = numpy.mean(y_true)
        self.error += numpy.absolute(y_utt - yhat_utt)
        self.total += 1


######################################################################
## Metrics that can be used as validation errors in chaipy.learning ##
######################################################################

VALIDATION_METRICS = {
    'mean_squared_error': mse, 'mse': mse,
    'root_mean_squared_error': rmse, 'rmse': rmse,
    'concordance_correlation_coefficient_error': ccc_err, 'ccc_err': ccc_err,
    'frame_error': FrameError, 'acc': FrameError,
    'frame_uar': FrameUnweightedAverageError, 'uar': FrameUnweightedAverageError,
    'utt_error': UttError, 'acc_utt': UttError,
    'utt_uar': UttUnweightedAverageError, 'uar_utt': UttUnweightedAverageError,
    'utt_mse': UttMeanSquaredError, 'mse_utt': UttMeanSquaredError,
    'utt_mabse': UttMeanAbsoluteError, 'mabse_utt': UttMeanAbsoluteError
}


########################################################################
## Metrics that can be used as objective functions in chaipy.learning ##
########################################################################

OBJECTIVE_FUNCTIONS = {
    'ccc': ccc_err_tensor, 'ccc_err': ccc_err_tensor,
    'ccc_multi': ccc_err_tensor_multi, 'ccc_err_multi': ccc_err_tensor_multi
}


#####################
## Generic metrics ##
#####################

def UAR(corr_lbls, pred_lbls):
    """ Compute UAR (Unweighted Average Recall). """
    return numpy.mean(comp_UARs(corr_lbls, pred_lbls)[0])


def comp_UARs(corr_lbls, pred_lbls):
    """ Compute recall rate for each class. Return (recall_rates, classes).
    """
    common.CHK_EQ(len(pred_lbls), len(corr_lbls))
    correct = {}
    count = {}
    for pred, corr in zip(pred_lbls, corr_lbls):
        if corr not in count:
            count[corr] = 0.0
            correct[corr] = 0.0
        count[corr] += 1
        correct[corr] += 1 if pred == corr else 0
    recall_rates = []
    classes = []
    for c in count.keys():
        recall_rates.append(correct[c] / count[c])
        classes.append(c)
    return (recall_rates, classes)


def AUC(corr_lbls, pred_scores, positive_lbl):
    """ Compute AUC (Area Under the Curve) """
    common.CHK_EQ(len(pred_scores), len(corr_lbls))
    fpr, tpr, _ = metrics.roc_curve(corr_lbls, pred_scores,
                                    pos_label=positive_lbl)
    return metrics.auc(fpr, tpr)

