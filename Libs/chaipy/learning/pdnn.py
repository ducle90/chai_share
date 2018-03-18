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

import os
import numpy
import collections
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import chaipy.common as common
import chaipy.io as io

# pdnn imports
from models.dnn import DNN
from models.srbm import SRBM
from io_func.model_io import string_2_array, _file2nnet, _nnet2file, _file2rbm
from utils.network_config import NetworkConfig, FactoredConfig
from utils.rbm_config import RBMConfig


class KLDNN(DNN):
    """
    KL-regularized DNN. See Yu et al. (ICASSP 2013).
    """
    def __init__(self, numpy_rng, base_dnn, rho=0.5, **kwargs):
        """
        Modify the objective function so it has this form:
            L = (1 - rho) * L' + rho * KL(y, y')
        Here L' is the original objective function, rho is the regularization
        weight. The larger rho is, the more weight we place on the base model.
        """
        DNN.__init__(self, numpy_rng, **kwargs)
        self.base_dnn = base_dnn
        self.rho = rho
        self.finetune_cost = (1 - rho) * self.finetune_cost
        self.kld_cost = -T.mean(T.sum(
            T.log(self.logLayer.p_y_given_x) * self.base_dnn.logLayer.p_y_given_x,
            axis=1
        ))
        self.finetune_cost += rho * self.kld_cost

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam*learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        if self.max_col_norm is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

        train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.base_dnn.x: train_set_x[index * batch_size:
                                             (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        valid_fn = self.build_validation_function(valid_shared_xy, batch_size)

        return train_fn, valid_fn


def get_dnn_cfg(dnn_fname):
    """ Construct a minimum required NetworkConfig given a model file """
    model_data = io.json_load(dnn_fname)
    cfg = NetworkConfig()
    cfg.hidden_layers_sizes = []
    i = 0
    while 'W{}'.format(i) in model_data:
        W_shape = string_2_array(model_data['W{}'.format(i)]).shape
        # Currently factored layer can only be the first hidden layer
        # TODO: change this!
        if i == 0:
            if 'side_W{}_0'.format(i) in model_data:
                factored_cfg = FactoredConfig()
                j = 0
                while 'side_b{}_{}'.format(i, j) in model_data:
                    assert 'side_W{}_{}'.format(i, j) in model_data
                    side_W_shape = string_2_array(model_data['side_W{}_{}'.format(i, j)]).shape
                    if j == 0:
                        factored_cfg.n_in_main = W_shape[0]
                        factored_cfg.n_in_side = side_W_shape[0]
                        # NOTE: this assumes that main and secondary features
                        # are disjoint, but this is not required by the model.
                        # TODO: find a way to relax this assumption.
                        cfg.n_ins = W_shape[0] + side_W_shape[0]
                    factored_cfg.side_layers.append(side_W_shape[1])
                    j += 1
                cfg.factored_cfg = factored_cfg
            else:
                cfg.n_ins = W_shape[0]
        if 'W{}'.format(i + 1) in model_data:
            cfg.hidden_layers_sizes.append(W_shape[1])
        else:
            cfg.n_outs = W_shape[1]
        i += 1
    return cfg


def init_dnn(dnn_cfg, numpy_rng=None, rho=0.0, base_dnn=None):
    """ Initialize a DNN given a DNN config """
    if numpy_rng is None:
        numpy_rng = numpy.random.RandomState()
    if rho != 0:
        return KLDNN(numpy_rng, base_dnn, rho=rho, cfg=dnn_cfg)
    return DNN(numpy_rng, cfg=dnn_cfg)


def load_dnn(dnn_fname, flat_start=False):
    """ Load a DNN from disk, automatically determining NetworkConfig.

    If `flat_start` is True, returned DNN will have the same architecture
    as the base DNN, but with randomly initialized weights instead.
    """
    cfg = get_dnn_cfg(dnn_fname)
    dnn = init_dnn(cfg)
    _file2nnet(dnn.layers, filename=dnn_fname)
    return dnn


def clone_dnn(dnn, numpy_rng=None):
    """ Return a copy of dnn """
    new_dnn = init_dnn(dnn.cfg, numpy_rng=numpy_rng)
    copy_dnn(dnn, new_dnn)
    return new_dnn


def copy_dnn(from_dnn, to_dnn):
    """ Copy weights from from_dnn to to_dnn. The number of layers must match.
    """
    common.CHK_EQ(len(from_dnn.layers), len(to_dnn.layers))
    for i in range(len(from_dnn.layers)):
        to_dnn.layers[i].W.set_value(from_dnn.layers[i].W.get_value())
        to_dnn.layers[i].b.set_value(from_dnn.layers[i].b.get_value())
    return to_dnn


def srbm_to_dnn(srbm, dnn):
    """ Copy weight parameters from SRBM to DNN. The number of SRBM layers
    must be smaller than or equal to the number of DNN hidden layers.
    """
    common.CHK_LT(len(srbm.rbm_layers), len(dnn.layers))
    for i in range(len(srbm.rbm_layers)):
        dnn.layers[i].W.set_value(srbm.rbm_layers[i].W.get_value())
        dnn.layers[i].b.set_value(srbm.rbm_layers[i].hbias.get_value())
    return dnn


def _train_sgd(train_fn, buf_train, dnn_cfg, shared_ds):
    """ Perform 1 iteration of SGD training.
    :type train_fn: theano.function
    :param train_fn: The pre-compiled training function to use

    :type buf_train: chaipy.data.temporal.BufferedTemporalData
    :param buf_train: The dataset to train on

    :type dnn_cfg: utils.network_config.NetworkConfig
    :param dnn_cfg: DNN config

    :type shared_ds: tuple (see BufferedTemporalData.make_shared)
    :param shared_ds: The shared dataset to use

    :rtype: list
    :return: List of mini-batch training errors
    """
    x, shared_x, y, shared_y = shared_ds[:4]
    train_error = []
    while True:
        # Load data chunk
        X = buf_train.read_next_chunk()
        if X is None:
            break
        # Set data
        frames = len(X[0])
        x[:frames] = X[0]
        shared_x.set_value(x[:frames], borrow=True)
        y[:frames] = X[1]
        shared_y.set_value(y[:frames], borrow=True)
        del X
        # Train
        io.log('Training on chunk {}'.format(buf_train.get_progress()))
        for batch_index in range(frames / dnn_cfg.batch_size):
            err = train_fn(index=batch_index, momentum=dnn_cfg.momentum,
                           learning_rate=dnn_cfg.lrate.get_rate())
            train_error.append(err)
    return train_error


def _validate(valid_fn, buf_valid, batch_size, shared_ds):
    """ Perform mini-batch validation.
    :type valid_fn: theano.function
    :param valid_fn: The pre-compiled validation function

    :type buf_valid: chaipy.data.temporal.BufferedTemporalData
    :param buf_valid: The dataset to validate on

    :type batch_size: int
    :param batch_size: Size of minibatch

    :type shared_ds: tuple (see BufferedTemporalData.make_shared)
    :param shared_ds: The shared dataset to use

    :rtype: list
    :return: List of minibatch validation errors
    """
    x, shared_x, y, shared_y = shared_ds[:4]
    valid_error = []
    while True:
        # Load data chunk
        X = buf_valid.read_next_chunk()
        if X is None:
            break
        # Set data
        frames = len(X[0])
        x[:frames] = X[0]
        shared_x.set_value(x[:frames], borrow=True)
        y[:frames] = X[1]
        shared_y.set_value(y[:frames], borrow=True)
        del X
        # Validate
        io.log('Validating on chunk {}'.format(buf_valid.get_progress()))
        for batch_index in range(frames / batch_size):
            valid_error.append(valid_fn(index=batch_index))
    return valid_error


def eval_dnn(dnn, buf_dataset, shared_ds=None):
    """ Return the mini-batch error rate on the given dataset.
    :type dnn: models.dnn.DNN
    :param dnn: The DNN to use for evaluation

    :type buf_dataset: chaipy.data.temporal.BufferedTemporalData
    :param buf_dataset: The dataset to evaluate on

    :type shared_ds: tuple (see BufferedTemporalData.make_shared)
    :param shared_ds: (optional) The shared dataset to use. If not set, will
        be automatically created from buf_dataset.

    :rtype: float
    :return: The mean mini-batch error rate (percentage)
    """
    if shared_ds is None:
        shared_ds = buf_dataset.make_shared()
    x, shared_x, y, shared_y = shared_ds[:4]
    # Compile validation function
    io.log('... getting the validation function')
    valid_fn = dnn.build_validation_function((shared_x, shared_y),
                                             batch_size=dnn.cfg.batch_size)
    io.log('Got it!')
    # Get error
    errors = _validate(valid_fn, buf_dataset, dnn.cfg.batch_size, shared_ds)
    return 100 * numpy.mean(errors)


def train_dnn(dnn, buf_train, buf_valid, shared_ds=None,
              save_dir=None, restore=False, pre_validate=False):
    """ Train DNN given a training and validation set.
    :type dnn: models.dnn.DNN
    :param dnn: The DNN to train

    :type buf_train: chaipy.data.temporal.BufferedTemporalData
    :param buf_train: The dataset to train on

    :type buf_valid: chaipy.data.temporal.BufferedTemporalData
    :param buf_valid: The dataset to validate on

    :type shared_ds: tuple (see BufferedTemporalData.make_shared)
    :param shared_ds: (optional) The shared dataset to use. If not set,
        will be set automatically using either buf_train or buf_valid,
        whichever has a bigger maximum partition size.

    :type save_dir: str
    :param save_dir: (optional) If not None, save the most recent intermediate
        model to this directory. We only keep the most recent model in this
        directory, except for the final model since we expect the caller of
        this function to save it manually.

    :type restore: bool
    :param restore: (optional) If True, restore parameters of the previous
        model if new validation error is higher than the lowest error thus
        far. This strategy is suitable for less stable learning.

    :type pre_validate: bool
    :param pre_validate: (optional) If True, do one validation iteration
        before training the model and use this value to bootstrap lrate.

    :rtype: tuple
    :return: (training errors, validation errors)
    """
    if shared_ds is None:
        if buf_train.max_partition_size() > buf_valid.max_partition_size():
            shared_ds = buf_train.make_shared()
        else:
            shared_ds = buf_valid.make_shared()
    if save_dir is not None and not os.path.exists(save_dir):
        os.make_dirs(save_dir, 0755)
    x, shared_x, y, shared_y = shared_ds[:4]
    # Compile finetuning function
    shared_xy = (shared_x, shared_y)
    io.log('... getting the finetuning functions')
    train_fn, valid_fn = \
            dnn.build_finetune_functions(shared_xy, shared_xy,
                                         batch_size=dnn.cfg.batch_size)
    io.log('Got them!')

    io.log('... finetuning the model')
    train_errs, valid_errs = [], []
    prev_params, prev_dparams = None, None
    # Do one preemptive validation iteration if necessary
    if pre_validate:
        train_errs.append(-1.0)
        io.log('** Pre-validate: training error {} (%)'.format(train_errs[-1]))
        valid_errs.append(100 * numpy.mean(
            _validate(valid_fn, buf_valid, dnn.cfg.batch_size, shared_ds)
        ))
        io.log('** Pre-validate: validation error {} (%)'.format(valid_errs[-1]))
        dnn.cfg.lrate.lowest_error = valid_errs[-1]
        if restore:
            prev_params = [p.get_value(borrow=True) for p in dnn.params]
            prev_dparams = [p.get_value(borrow=True) for p in dnn.delta_params]
    # Start training
    while dnn.cfg.lrate.get_rate() != 0:
        # One epoch of SGD training
        train_errs.append(100 * numpy.mean(
            _train_sgd(train_fn, buf_train, dnn.cfg, shared_ds)
        ))
        io.log('** Epoch {}, lrate {}, training error {} (%)'.format(
            dnn.cfg.lrate.epoch, dnn.cfg.lrate.get_rate(), train_errs[-1]
        ))
        valid_errs.append(100 * numpy.mean(
            _validate(valid_fn, buf_valid, dnn.cfg.batch_size, shared_ds)
        ))
        io.log('** Epoch {}, lrate {}, validation error {} (%)'.format(
            dnn.cfg.lrate.epoch, dnn.cfg.lrate.get_rate(), valid_errs[-1]
        ))
        prev_error = dnn.cfg.lrate.lowest_error
        dnn.cfg.lrate.get_next_rate(current_error=valid_errs[-1])
        io.log('**** Updated lrate: {}'.format(dnn.cfg.lrate.get_rate()))
        # Restore model parameters if necessary
        if restore:
            if valid_errs[-1] < prev_error:
                prev_params = [p.get_value(borrow=True) for p in dnn.params]
                prev_dparams = [p.get_value(borrow=True) for p in dnn.delta_params]
            elif prev_params is None:
                io.log('**WARN** error increased but no prev_params to restore!')
            elif dnn.cfg.lrate.epoch <= dnn.cfg.lrate.min_epoch_decay_start:
                io.log('** Only {} training epoch, need at least {} to restore **'.format(
                    dnn.cfg.lrate.epoch - 1, dnn.cfg.lrate.min_epoch_decay_start
                ))
            else:
                io.log('** Restoring params of previous best model **')
                for cp, pp in zip(dnn.params, prev_params):
                    cp.set_value(pp, borrow=True)
                for cdp, pdp in zip(dnn.delta_params, prev_dparams):
                    cdp.set_value(pdp, borrow=True)
                idx = numpy.argmin(valid_errs)
                io.log('** Restored: train err = {}, valid err = {} **'.format(
                    train_errs[idx], valid_errs[idx]
                ))
        # Save intermediate model
        if save_dir is not None:
            curr_epoch = dnn.cfg.lrate.epoch - 1
            prev_epoch = curr_epoch - 1
            curr_fname = os.path.join(save_dir, '{}.dnn'.format(curr_epoch))
            prev_fname = os.path.join(save_dir, '{}.dnn'.format(prev_epoch))
            if dnn.cfg.lrate.get_rate() != 0:
                _nnet2file(dnn.layers, filename=curr_fname)
            if os.path.exists(prev_fname):
                os.remove(prev_fname)
    # If restoring, make sure the final err is also the best err
    if restore and valid_errs[-1] != numpy.min(valid_errs):
        idx = numpy.argmin(valid_errs)
        train_errs.append(train_errs[idx])
        valid_errs.append(valid_errs[idx])
    return (train_errs, valid_errs)


def get_srbm_cfg(srbm_fname):
    """ Construct a minimum required RBMConfig given a model file """
    model_data = io.json_load(srbm_fname)
    cfg = RBMConfig()
    cfg.hidden_layers_sizes = []
    i = 0
    while 'W{}'.format(i) in model_data:
        W_shape = string_2_array(model_data['W{}'.format(i)]).shape
        if i == 0:
            cfg.n_ins = W_shape[0]
        cfg.hidden_layers_sizes.append(W_shape[1])
        i += 1
    return cfg


def init_srbm(rbm_cfg, numpy_rng=None):
    """ Initialize a SRBM given a RBM config """
    if numpy_rng is None:
        numpy_rng = numpy.random.RandomState()
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    # Following pdnn, initialize a parallel DNN and use it to initialize SRBM
    dnn_cfg = NetworkConfig()
    dnn_cfg.n_ins = rbm_cfg.n_ins
    dnn_cfg.hidden_layers_sizes = rbm_cfg.hidden_layers_sizes
    dnn_cfg.n_outs = rbm_cfg.n_outs
    dnn = DNN(numpy_rng, theano_rng=theano_rng, cfg=dnn_cfg)
    return SRBM(numpy_rng, theano_rng=theano_rng, cfg=rbm_cfg, dnn=dnn)


def load_srbm(srbm_fname):
    """ Load a SRBM from disk, automatically determining RBMConfig """
    cfg = get_srbm_cfg(srbm_fname)
    srbm = init_srbm(cfg)
    _file2rbm(srbm.rbm_layers, filename=srbm_fname)
    return srbm


def copy_srbm(from_srbm, to_srbm):
    """ Copy weights from from_srbm to to_srbm. The number of layers in
    from_srbm must be smaller than or equal to the number of layers in to_srbm.
    """
    common.CHK_LE(len(from_srbm.rbm_layers), len(to_srbm.rbm_layers))
    for i in range(len(from_srbm.rbm_layers)):
        from_layer, to_layer = from_srbm.rbm_layers[i], to_srbm.rbm_layers[i]
        to_layer.W.set_value(from_layer.W.get_value())
        to_layer.hbias.set_value(from_layer.hbias.get_value())
        to_layer.vbias.set_value(from_layer.vbias.get_value())
    return to_srbm


def train_srbm(srbm, buf_train, layer, weight_cost=0.0002, cd_iter=1,
               stop_thresh=0.001, shared_ds=None):
    """ Train a specific layer within the SRBM.
    :type srbm: models.srbm.SRBM
    :param srbm: The SRBM to train

    :type buf_train: chaipy.data.temporal.BufferedTemporalData
    :param buf_train: The dataset to train the SRBM on

    :type layer: int
    :param layer: Index of the layer to train

    :type weight_cost: float
    :param weight_cost: (optional) L2 regularization coefficient

    :type cd_iter: int
    :param cd_iter: (optional) The number of CD iterations to perform

    :type stop_thresh: float
    :param stop_thresh: (optional) Stop pretraining when relative change in
        reconstruction cost is smaller than this threshold.

    :type shared_ds: tuple (see BufferedTemporalData.make_shared)
    :param shared_ds: (optional) The shared dataset to use. If not set, will
        be created automatically using buf_train.

    :rtype: models.srbm.SRBM
    :return: The trained SRBM
    """
    common.CHK_LT(layer, len(srbm.rbm_layers))
    if shared_ds is None:
        shared_ds = buf_train.make_shared(use_labels=False)
    x, shared_x = shared_ds[:2]
    # Compile pretraining function
    io.log('... getting the pretraining function for RBM layer {}'.format(layer))
    pretrain_fn = srbm.pretraining_functions(train_set_x=shared_x, k=cd_iter,
            batch_size=srbm.cfg.batch_size, weight_cost=weight_cost,
            layers=[layer])[0]
    io.log('Got it!')
    # Determine the learning rate
    if srbm.rbm_layers[layer].is_gbrbm():
        pretrain_lr = srbm.cfg.gbrbm_learning_rate
    else:
        pretrain_lr = srbm.cfg.learning_rate
    io.log('Pretraining learning rate = {}'.format(pretrain_lr))

    io.log('... pretraining the layer')
    io.log('(r_c = reconstruction cost, fe_c = approximate free energy function)')
    # Do training
    epoch = 0
    momentum = srbm.cfg.initial_momentum
    io.log('Momentum = {}'.format(momentum))
    prev_r_c = None
    r_cs, fe_cs = [], []  # Mini-batch reconstruction and free-energy costs
    while True:
        # Load data chunk
        X = buf_train.read_next_chunk()
        if X is not None:
            frames = len(X[0])
            x[:frames] = X[0]
            shared_x.set_value(x[:frames], borrow=True)
            del X
            io.log('Training on chunk {}'.format(buf_train.get_progress()))
            for batch_index in range(frames / srbm.cfg.batch_size):
                r_c, fe_c = pretrain_fn(index=batch_index, lr=pretrain_lr,
                                        momentum=momentum)
                r_cs.append(r_c)
                fe_cs.append(fe_c)
        else:
            epoch += 1
            r_c = numpy.mean(r_cs)
            fe_c = numpy.mean(fe_cs)
            io.log('Finished epoch {}, r_c {}, fe_c {}'.format(epoch, r_c, fe_c))
            change = None if prev_r_c is None else (prev_r_c - r_c) / prev_r_c
            io.log('prev_r_c {}, r_c {}, change {}'.format(prev_r_c, r_c, change))
            if change is not None and abs(change) < stop_thresh and \
                    epoch > srbm.cfg.initial_momentum_epoch:
                io.log('** FINISHED! (stop_thresh = {})'.format(stop_thresh))
                break
            else:
                if epoch == srbm.cfg.initial_momentum_epoch:
                    momentum = srbm.cfg.final_momentum
                    io.log('Updated momentum = {}'.format(momentum))
                prev_r_c = r_c
                r_cs, fe_cs = [], []
    return srbm

