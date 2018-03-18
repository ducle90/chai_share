# Copyright 2016    Duc Le  University of Michigan
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
import os
import inspect

import chaipy.common as common
import chaipy.io as io
from chaipy.common.metrics import AbstractMetric, KerasMetric, \
                                  VALIDATION_METRICS, OBJECTIVE_FUNCTIONS

# keras imports
from keras.optimizers import adam


def init_metric(m):
    if type(m) == str:
        common.CHK_VALS(m, VALIDATION_METRICS.keys())
        m = VALIDATION_METRICS[m]
    if inspect.isfunction(m):
        return KerasMetric(m)
    assert issubclass(m, AbstractMetric)
    return m()


def init_loss(l):
    if type(l) == str:
        if l not in OBJECTIVE_FUNCTIONS:
            return l
        l = OBJECTIVE_FUNCTIONS[l]
    assert inspect.isfunction(l), 'Loss must be a string or function'
    return l


def validate_rnn(model, buf_valid, metrics=['acc'], stateful=False,
                 report_interval=20):
    """
    Evaluate against a dataset. Return a list of values, one for each metric.

    `metrics` must be a list.
    """
    metrics = [init_metric(m) for m in metrics]
    chunks_read = 0
    while True:
        # Load data chunk
        X = buf_valid.read_next_chunk()
        if X is None:
            break
        report = chunks_read % report_interval == 0
        chunks_read += 1
        # Validate
        valid_Xs, valid_ys, valid_eobs, utt_indices = X
        if report:
            io.log('Validating on chunk {} ({} utts, max dur {})'.format(
                buf_valid.get_progress(), len(utt_indices), valid_Xs[0].shape[1]
            ))
        for valid_X, valid_y, valid_eob in zip(valid_Xs, valid_ys, valid_eobs):
            preds = model.predict(
                valid_X, batch_size=len(utt_indices), verbose=0
            )
            for i in range(len(valid_eob)):
                y_true = valid_y[i][buf_valid.get_delay():valid_eob[i]]
                y_pred = preds[i][buf_valid.get_delay():valid_eob[i]]
                for m in metrics:
                    m.accum(y_true, y_pred)
        if stateful:
            model.reset_states()
    return [m.eval() for m in metrics]


def _train(model, buf_train, metrics=['acc'], stateful=False,
           loss='sparse_categorical_crossentropy', report_interval=20):
    train_metrics = []
    for _ in metrics:
        train_metrics.append([])
    chunks_read = 0
    while True:
        # Load data chunk
        X = buf_train.read_next_chunk()
        if X is None:
            break
        report = chunks_read % report_interval == 0
        chunks_read += 1
        # Train
        train_Xs, train_ys, _, utt_indices = X
        if report:
            io.log('Training on chunk {} ({} utts, max dur {})'.format(
                buf_train.get_progress(), len(utt_indices), train_Xs[0].shape[1]
            ))
        for train_X, train_y in zip(train_Xs, train_ys):
            if loss == 'sparse_categorical_crossentropy':
                train_y = numpy.expand_dims(train_y, -1)
            history = model.fit(
                train_X, train_y, batch_size=len(utt_indices),
                nb_epoch=1, verbose=0
            )
            # Save metrics
            for tm, m in zip(train_metrics, metrics):
                tm.append(history.history[m][0])
        if report:
            io.log('...last metrics = {}'.format(
                [train_metrics[i][-1] for i in range(len(metrics))]
            ))
        if stateful:
            model.reset_states()
    for i in range(len(metrics)):
        train_metrics[i] = numpy.mean(train_metrics[i])
        if metrics[i] == 'acc':   # We want training error in percentage
            train_metrics[i] = 100 * (1.0 - train_metrics[i])
    return train_metrics


def train_rnn(model, buf_train, buf_valid, lrate, **kwargs):
    """
    Train RNN given a training and a single validation set.

    :rtype: tuple
    :return: (training errors, validation errors)
    """
    train_errs, valid_errs = train_rnn_v2(
        model, buf_train, [buf_valid], lrate, **kwargs
    )
    return (train_errs, valid_errs[0])


def train_rnn_v2(model, buf_train, buf_valids, lrate, validate_on=0,
                 optimizer=None, save_dir=None, pre_validate=False,
                 restore=False, loss='sparse_categorical_crossentropy',
                 metrics=['acc'], validate_metrics=None, stateful=False):
    """
    Train RNN given a training and multiple validation sets.

    :type model: Keras model
    :param model: The model to train

    :type buf_train: chaipy.data.temporal.BufferedUttData
    :param buf_train: The dataset to train on

    :type buf_valids: list of chaipy.data.temporal.BufferedUttData
    :param buf_valids: The datasets to validate on

    :type lrate: utils.learn_rates.LearningRate
    :param lrate: Object to control learning rate decay

    :type validate_on: int
    :param validate_on: (optional) Which validation set to use for learning
        rate schedule. By default, use the first validation set.

    :type optimizer: keras.optimizers.Optimizer
    :param optimizer: (optional) Optimizer to use. If not set, use Adam.

    :type save_dir: str
    :param save_dir: (optional) If not None, save the most recent intermediate
        model to this directory. We only keep the most recent model in this
        directory, except for the final model since we expect the caller of
        this function to save it manually. The caller is responsible for the
        creation and deletion of this directory. The code does not create and
        delete the directory automatically.

    :type pre_validate: bool
    :param pre_validate: (optional) If True, do one validation iteration
        before training the model and use this value to bootstrap lrate.

    :type restore: bool
    :param restore: (optional) If True, restore parameters of the previous model
        if new validation error is higher than the lowest error thus far. This
        strategy is suitable for less stable learning.

    :type loss: str or keras loss function
    :param loss: (optional) Loss function for training. Accepted types are:
            * str: must be a key in `chaipy.common.metrics.OBJECTIVE_FUNCTIONS`
                   or one recognized by keras.
            * function: a symbolic function with signature `fn(y_true, y_pred)`

    :type metrics: list of str
    :param metrics: (optional) Metrics to report when showing training progress.
        Must use the exact Keras name defined by the metric function! For
        example, use 'mean_squared_error' instead of 'mse'; use 'acc' instead
        of 'accuracy'. If multiple metrics are specified, the first metric is
        used to log training errors.

    :type validate_metrics: list
    :param validate_metrics: (optional) Which metrics to use for validation.
        If unspecified, use training metrics. If multiple metrics are specified,
        the first metric is used for actual validation. Accepted types are:
            * str: must be a key in `chaipy.common.metrics.VALIDATION_METRICS`
            * function: must have the signature `fn(y_true, y_pred)`
            * class: must be a subclass of `chaipy.common.metrics.AbstractMetric`

    :type stateful: bool
    :param stateful: (optional) Network is stateful.

    :rtype: tuple
    :return: (training errors, [validation errors])
    """
    common.CHK_EQ(type(buf_valids), list)
    if optimizer is None:
        optimizer=adam(lr=lrate.get_rate())
    loss = init_loss(loss)
    if validate_metrics is None:
        validate_metrics = metrics
    io.log('... finetuning the model')
    train_errs, valid_errs = [], []
    for _ in range(len(buf_valids)):
        valid_errs.append([])
    prev_weights = None
    # Do one premptive validation iteration if necessary
    if pre_validate:
        train_errs.append(-1)
        for i in range(len(buf_valids)):
            all_valid_errs = validate_rnn(
                model, buf_valids[i], metrics=validate_metrics, stateful=stateful
            )
            if len(validate_metrics) > 1:
                io.log('** Pre-validate: all validation errors ({}) {}'.format(
                    i, all_valid_errs
                ))
            valid_errs[i].append(all_valid_errs[0])
            io.log('** Pre-validate: validation error ({}) {}'.format(
                i, valid_errs[i][-1]
            ))
        lrate.lowest_error = valid_errs[validate_on][-1]
        if restore:
            prev_weights = [l.get_weights() for l in model.layers]
    # Start training
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    while lrate.get_rate() != 0:
        model.optimizer.lr.set_value(lrate.get_rate())
        io.log('*** Optimizer state: {}'.format(model.optimizer.get_config()))
        # One epoch of training
        all_train_errs = _train(
            model, buf_train,
            metrics=metrics, stateful=stateful, loss=loss
        )
        if len(metrics) > 1:
            io.log('** Epoch {}, lrate {}, all training errors {}'.format(
                lrate.epoch, lrate.get_rate(), all_train_errs
            ))
        train_errs.append(all_train_errs[0])
        io.log('** Epoch {}, lrate {}, training error {}'.format(
            lrate.epoch, lrate.get_rate(), train_errs[-1]
        ))
        for i in range(len(buf_valids)):
            all_valid_errs = validate_rnn(
                model, buf_valids[i], metrics=validate_metrics, stateful=stateful
            )
            if len(validate_metrics) > 1:
                io.log('** Epoch {}, lrate {}, all validation errors ({}) {}'.format(
                    lrate.epoch, lrate.get_rate(), i, all_valid_errs
                ))
            valid_errs[i].append(all_valid_errs[0])
            io.log('** Epoch {}, lrate {}, validation error ({}) {}'.format(
                lrate.epoch, lrate.get_rate(), i, valid_errs[i][-1]
            ))
        prev_error = lrate.lowest_error
        lrate.get_next_rate(current_error=valid_errs[validate_on][-1])
        io.log('**** Updated lrate: {}'.format(lrate.get_rate()))
        # Restore model weights if necessary
        if restore:
            if valid_errs[validate_on][-1] < prev_error:
                prev_weights = [l.get_weights() for l in model.layers]
            elif prev_weights is None:
                io.log('**WARN** error increased but no prev_weights to restore!')
            elif lrate.epoch <= lrate.min_epoch_decay_start:
                io.log('** Only {} training epoch, need at least {} to restore **'.format(
                    lrate.epoch - 1, lrate.min_epoch_decay_start
                ))
            else:
                io.log('** Restoring params of previous best model **')
                for l, lw in zip(model.layers, prev_weights):
                    l.set_weights(lw)
                idx = numpy.argmin(valid_errs[validate_on])
                io.log('** Restored: train err = {}, valid errs = {} **'.format(
                    train_errs[idx], [valid_errs[i][idx] for i in range(len(buf_valids))]
                ))
        # Save intermediate model
        if save_dir is not None:
            curr_epoch = lrate.epoch - 1
            prev_epoch = curr_epoch - 1
            curr_fname = os.path.join(save_dir, '{}'.format(curr_epoch))
            prev_fname = os.path.join(save_dir, '{}'.format(prev_epoch))
            if lrate.get_rate() != 0:
                io.json_save('{}.json'.format(curr_fname), model.to_json())
                model.save_weights('{}.weights'.format(curr_fname))
            for suffix in ['json', 'weights']:
                if os.path.exists('{}.{}'.format(prev_fname, suffix)):
                    os.remove('{}.{}'.format(prev_fname, suffix))
    # If restoring, make sure the final err is also the best err
    if restore and valid_errs[validate_on][-1] != numpy.min(valid_errs[validate_on]):
        idx = numpy.argmin(valid_errs[validate_on])
        train_errs.append(train_errs[idx])
        for i in range(len(buf_valids)):
            valid_errs[i].append(valid_errs[i][idx])
    return (train_errs, valid_errs)

