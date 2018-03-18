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

import sys
import os
import shutil
import numpy as np
from collections import OrderedDict
import itertools

import chaipy.common as common
import chaipy.io as io
import chaipy.learning.rnn as learning_rnn
import chaipy.learning.mtrnn as learning_mtrnn
from chaipy.kaldi import ivector_ark_read
from chaipy.data import is_incremental
from chaipy.data.temporal import TemporalData, MultiLabelTemporalData, \
                                 BufferedUttData, AugmentedBufferedUttData, \
                                 BufferedVarUttData, AugmentedBufferedVarUttData
from chaipy.data.incremental import DataGenerator, IncrementalDataBuffer

# keras imports
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, SimpleRNN, GRU, Dropout
from keras.layers import Dense, Bidirectional, TimeDistributed, Lambda
from keras.optimizers import sgd, adam, rmsprop
from keras.regularizers import WeightRegularizer
# pdnn imports
from utils.learn_rates import LearningRateExpDecay, LearningMinLrate

# Accepted layer types
LAYERS = {
    'rnn': SimpleRNN,
    'gru': GRU,
    'lstm': LSTM
}
# Accepted optimizers
OPTIMIZERS = {
    'sgd': sgd,
    'adam': adam,
    'rmsprop': rmsprop
}
# Accepted learning rate schedulers
LRATES = {
    'exp_decay': LearningRateExpDecay,
    'step_decay': LearningMinLrate
}
# Accepted tasks
TASKS = {
    'classification',
    'regression'
}
# Useful constants
FLT_MAX = np.finfo(np.float32).max


def init_reg(l1, l2):
    if l1 == 0 and l2 == 0:
        return None
    return WeightRegularizer(l1=l1, l2=l2)


def init_optimizer(opt_type, lr, clip_grad=None):
    kwargs = {} if clip_grad is None else { 'clipvalue': clip_grad }
    return opt_type(lr=lr, **kwargs)


def parse_lrate_json(json, lrate):
    lrate.rate = json['lrate']
    lrate.epoch = json['epoch']
    if 'decay' in json:
        lrate.decay = json['decay']
    return lrate


def init_lrate(cls, init_lr, min_lr, depoch, dstart, dstop,
               max_epoch=None, resume_json=None):
    io.log('.....using {}'.format(cls))
    lrate = cls(
        start_rate=init_lr, min_lrate_stop=min_lr,
        min_epoch_decay_start=depoch, min_derror_decay_start=dstart,
        min_derror_stop=dstop, max_epoch=max_epoch
    )
    if resume_json is not None:
        io.log('Resuming learning rate from {}'.format(resume_json))
        lrate = parse_lrate_json(resume_json, lrate)
    return lrate


def init_rnn(n_ins, n_outs, layer_type, layer_size, num_layers, bidirectional,
             dropout=0.0, l1_reg=0.0, l2_reg=0.0, ext_weights_fname=None,
             non_trainable=[], input=None):
    # Input node
    if input is None:
        input = Input(batch_shape=(None, None, n_ins), name='input')
    x = input
    # Recurrent hidden layers
    for i in range(num_layers):
        trainable = i not in non_trainable
        if not trainable:
            io.log('Hidden layer {} frozen'.format(i))
        recurrent_layer = layer_type(
            layer_size, return_sequences=True, consume_less='gpu',
            W_regularizer=init_reg(l1_reg, l2_reg),
            U_regularizer=init_reg(l1_reg, l2_reg),
            trainable=trainable
        )
        if bidirectional:
            recurrent_layer = Bidirectional(
                recurrent_layer, merge_mode='concat'
            )
        x = recurrent_layer(x)
        if dropout != 0:
            x = Dropout(dropout)(x)
    # Output layers
    if type(n_outs) != list:
        io.log('Single-task: {}'.format(n_outs))
        n_outs = [n_outs]   # keras will treat this as single-task model
    else:
        io.log('Multi-task: {}'.format(n_outs))
    trainable = num_layers not in non_trainable
    if not trainable:
        io.log('Output layer(s) frozen')
    output_layers = []
    for i in range(len(n_outs)):
        if n_outs[i] != 1:  # Softmax for classification
            output_layers.append(TimeDistributed(Dense(
                n_outs[i], activation='softmax', trainable=trainable,
                W_regularizer=init_reg(l1_reg, l2_reg)
            ))(x))
        else:               # Flatten output for regression
            y = TimeDistributed(Dense(
                n_outs[i], activation='linear', trainable=trainable,
                W_regularizer=init_reg(l1_reg, l2_reg)
            ))(x)
            output_layers.append(Lambda(
                lambda m: K.squeeze(m, axis=2), output_shape=lambda s: (s[0], s[1])
            )(y))
    model = Model(input=input, output=output_layers)
    io.log('Model architecture: {}'.format(model.layers))
    # Load existing weights if possible
    if ext_weights_fname is not None:
        io.log('Loading existing weights from {}'.format(ext_weights_fname))
        model.load_weights(ext_weights_fname)
    return model


def lname(layer_type, bidirectional):
    if bidirectional:
        return 'b{}'.format(layer_type)
    return layer_type


def save_records(fname, records):
    records_content = []
    for model_name in records:
        content = records[model_name]
        records_content.append('{} {}'.format(
            model_name, ' '.join([str(x) for x in content])
        ))
    io.write_lines(fname, records_content)


class FinetuneRNNEngine(object):

    def run(self):
        self.parse_args()
        self.init_data()
        self.pre_execute()
        params_title, params_iter = self.params_iter()
        for params in params_iter:
            params_map = OrderedDict()
            for title, param in zip(params_title, params):
                params_map[title] = param
            self.step(**params_map)
        self.post_execute()


    #####################
    ## Argument parser ##
    #####################

    def parse_args(self):
        parser = common.init_argparse(self._desc())
        self._register_custom_types(parser)
        self._main_args(parser)
        self._validation_args(parser)
        self._data_args(parser)
        self._model_args(parser)
        self._training_args(parser)
        self.args = parser.parse_args()
        self._check_args()

    def _desc(self):
        """ Title of program to print in help text.
        """
        return 'Finetune RNN model. Supports classification and regression.'

    def _register_custom_types(self, parser):
        """ Register custom types for arguments.
        """
        parser.register('type', 'bool', lambda x: x.lower() in ['yes', 'true'])
        return parser

    def _main_args(self, parser):
        """ Setup main arguments.
        """
        group = parser.add_argument_group(title='Main Arguments')
        group.add_argument('train_scp', help='scp of training data.')
        group.add_argument('train_labels', help='Labels of training data.')
        group.add_argument('output_dir', help='Output directory to save results.')
        return group

    def _validation_args(self, parser):
        """ Setup arguments for validation sets.
        """
        group = parser.add_argument_group(title='Validation Setup')
        group.add_argument('--valid-scp', nargs='+', default=[],
                           help='scp of validation data (must specify ' + \
                                'at least one, can have multiple).')
        group.add_argument('--valid-labels', nargs='+', default=[],
                           help='Labels of validation data (must specify ' + \
                                'at least one, can have multiple).')
        group.add_argument('--validate-on', type=int, default=0,
                           help='Which validation set to validate on.')
        return group

    def _data_args(self, parser):
        """ Setup arguments for data provider.
        """
        group = parser.add_argument_group(title='Data Provider Configuration')
        group.add_argument('--task', default='classification',
                           help='Target task ({}).'.format(list(TASKS)))
        group.add_argument('--use-var-utt', type='bool', default=True,
                           help='Use variable utterances per minibatch. ' + \
                                'Can reduce GPU transfer for large data.')
        group.add_argument('--max-frames', type=int, default=None,
                           help='Maximum utterance length in frames. ' + \
                                'If set, omit utterances longer than this.')
        group.add_argument('--nutts', type=int, default=10,
                           help='Number of utterances to bundle per batch.')
        group.add_argument('--delay', type=int, default=0,
                           help='Delay output by this many frames.')
        group.add_argument('--context', type=int, default=5,
                           help='Number of context frames to use each side.')
        group.add_argument('--ivectors', default=None,
                           help='Path to i-vector ark to consider.')
        group.add_argument('--num-classes', type=int, default=None,
                           help='Number of classes (classification only). ' + \
                                'If not set, use max value in `train_labels`.')
        return group

    def _model_args(self, parser):
        """ Setup arguments for model.
        """
        group = parser.add_argument_group(title='Model Configuration')
        group.add_argument('--layer-type', nargs='+', default=['lstm'],
                           help='Layer type to try ({}).'.format(LAYERS.keys()))
        group.add_argument('--layer-size', type=int, nargs='+', default=[600],
                           help='Size of hidden layers to try.')
        group.add_argument('--num-layers', type=int, nargs='+', default=[3],
                           help='Number of hidden layers to try.')
        group.add_argument('--bidirectional', type='bool', nargs='+', default=[True],
                           help='Whether or not to use bi-directional wrapper.')
        group.add_argument('--dropout', type=float, nargs='+', default=[0.0],
                           help='Dropout to apply to output of hidden layers.')
        group.add_argument('--l1-reg', type=float, nargs='+', default=[0.0],
                           help='L1 regularization weights to try.')
        group.add_argument('--l2-reg', type=float, nargs='+', default=[0.0],
                           help='L2 regularization weights to try.')
        group.add_argument('--ext-weights', default=None,
                           help='Bootstrap model with existing weights. ' + \
                                'Model architecture must stay constant if ' + \
                                'this option is used.')
        group.add_argument('--non-trainable', nargs='+', type=int, default=[],
                            help='Non-trainable layers. 0 is the 1st layer.' + \
                                 'Beware of changing model architecture ' + \
                                 'when using this option.')
        return group

    def _training_args(self, parser):
        """ Setup arguments for training.
        """
        group = parser.add_argument_group(title='Training Configuration')
        group.add_argument('--optimizer', default='adam',
                           help='Type of optimizer ({}).'.format(OPTIMIZERS.keys()))
        group.add_argument('--clip-grad', type=float,
                           help='Clip gradients whose absolute value exceeds this.')
        group.add_argument('--lrate', default='exp_decay',
                           help='Learning rate scheduler ({}).'.format(LRATES.keys()))
        group.add_argument('--init-lr', type=float, default=1e-3,
                           help='Initial learning rate.')
        group.add_argument('--min-lr', type=float, default=1e-5,
                           help='Min learning rate.')
        group.add_argument('--depoch', type=int, default=1,
                           help='Minimum training epochs before decaying.')
        group.add_argument('--dstart', type=float, default=0.0,
                           help='Start decay if error diff is smaller than this.')
        group.add_argument('--dstop', type=float, default=0.0,
                           help='Stop training if error diff is smaller than this.')
        group.add_argument('--max-epoch', type=int, default=None,
                           help='Maximum number of epochs to train.')
        group.add_argument('--save-all', type='bool', default=False,
                           help='If True, save all models, not just the best one.')
        return group

    def _check_args(self):
        """ Perfunctory argument checks.
        """
        args = self.args
        common.CHK_GE(len(args.valid_scp), 1)
        common.CHK_EQ(len(args.valid_scp), len(args.valid_labels))
        common.CHK_VALS(args.task, TASKS)
        for layer_type in args.layer_type:
            common.CHK_VALS(layer_type, LAYERS.keys())
        common.CHK_VALS(args.optimizer, OPTIMIZERS.keys())
        common.CHK_VALS(args.lrate, LRATES.keys())


    #####################
    ## Initialize data ##
    #####################

    def init_data(self):
        io.log('Initializing data...')
        self._init_ivectors()
        self._init_base_data()
        self._init_buf_data()

    def _init_ivectors(self):
        """ Add i-vectors if applicable.
        """
        args = self.args
        self.ivectors = None
        if args.ivectors is not None:
            io.log('Loading i-vectors from {}'.format(args.ivectors))
            self.ivectors = ivector_ark_read(args.ivectors)

    def _init_base_data(self):
        """ Setup base data objects.
        """
        args = self.args
        labels_dtype = np.int32 if args.task == 'classification' else np.float32
        self.train = self._load_base_data(
            args.train_scp, args.train_labels, labels_dtype
        )
        self.valids = []
        for valid_scp, valid_labels in zip(args.valid_scp, args.valid_labels):
            self.valids.append(self._load_base_data(
                valid_scp, valid_labels, labels_dtype
            ))

    def _load_base_data(self, scp_fname, labels_fname, labels_dtype):
        """ Load a single base dataset.
        """
        args = self.args
        if is_incremental(scp_fname):
            scps = io.read_lines(scp_fname)
        else:
            scps = [scp_fname]
        data_gens = []
        for scp in scps:
            data_gens.append(DataGenerator(
                TemporalData.from_kaldi,
                scp=scp, alipdf=labels_fname, num_pdfs=args.num_classes,
                context=args.context, padding='replicate',
                utt_feats_dict=self.ivectors, labels_dtype=labels_dtype
            ))
        return data_gens

    def _init_buf_data(self):
        """ Setup data objects for buffering.
        """
        args = self.args
        buf_cls = BufferedUttData
        if args.use_var_utt:
            buf_cls = BufferedVarUttData
        self.buf_train = IncrementalDataBuffer(
            self.train, buf_cls, shuffle_gens=True,
            max_frames=args.max_frames, nutts=args.nutts,
            delay=args.delay, shuffle=True
        )
        self.buf_valids = []
        for valid in self.valids:
            self.buf_valids.append(IncrementalDataBuffer(
                valid, buf_cls, shuffle_gens=False,
                max_frames=args.max_frames, nutts=args.nutts,
                delay=args.delay, shuffle=False
            ))


    #################################
    ## Preparation for experiments ##
    #################################

    def pre_execute(self):
        # Hack to fix recursion depth exceeded
        sys.setrecursionlimit(1000000)
        io.log('args: {}'.format(vars(self.args)))
        self._init_output_dir()
        self._init_records()
        self._init_best_err()

    def _init_output_dir(self):
        """ Setup output directory.
        """
        io.log('Saving experiment results to {}'.format(self.args.output_dir))
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, 0755)

    def _init_records(self):
        """ Setup result caching for parameter combinations. Load existing
        results from disk if possible. Each line will look like this:
            <model_name> <train_err> <valid_err> [<valid_err> ...]
        """
        self.records_fname = os.path.join(self.args.output_dir, 'summary.txt')
        self.records = OrderedDict()
        if os.path.exists(self.records_fname):
            io.log('Loading existing records from {}'.format(self.records_fname))
            self.records = io.dict_read(
                self.records_fname, ordered=True, lst=True, fn=float
            )

    def _init_best_err(self):
        """ Determine the best model thus far.
        """
        self.best_model, self.best_err = None, FLT_MAX
        for model_name in self.records:
            valid_errs = self.records[model_name][1:]
            if valid_errs[self.args.validate_on] < self.best_err:
                self.best_model = model_name
                self.best_err = valid_errs[self.args.validate_on]
        io.log('Best existing model: {}, Best existing err: {}'.format(
            self.best_model, self.best_err
        ))


    ##########################
    ## Main experiment code ##
    ##########################

    def params_iter(self):
        args = self.args
        params_title = [
            'layer_type', 'layer_size', 'num_layers',
            'bidirectional', 'dropout', 'l1_reg', 'l2_reg'
        ]
        params_iter = itertools.product(
            args.layer_type, args.layer_size, args.num_layers,
            args.bidirectional, args.dropout, args.l1_reg, args.l2_reg
        )
        return (params_title, params_iter)

    def step(self, **kwargs):
        model_name = self._model_name(**kwargs)
        if self._ext_model(model_name):
            return
        io.log('--- {} ---'.format(model_name))
        model_dir = self._init_model_dir(model_name)
        resume_json = self._init_resume_json(model_dir)
        model = self._init_model(resume_json=resume_json, **kwargs)
        lrate = self._init_lrate(resume_json=resume_json, **kwargs)
        optimizer = self._init_optimizer(lrate=lrate, **kwargs)
        results = self._train(
            model=model, lrate=lrate, optimizer=optimizer,
            model_name=model_name, model_dir=model_dir,
            resume_json=resume_json,
            **kwargs
        )
        self._finalize(model, model_name, model_dir, results)
        # Clean up
        shutil.rmtree(model_dir)
        del model

    def _model_name(self, layer_type, layer_size, num_layers,
                    bidirectional, dropout, l1_reg, l2_reg, **kwargs):
        """ Return a string that identifies this model combination.
        """
        return '{}_{}x{}+dropout_{}+l1_{}+l2_{}'.format(
            lname(layer_type, bidirectional), num_layers, layer_size,
            dropout, l1_reg, l2_reg
        )

    def _ext_model(self, model_name):
        """ Check if model configuration has already been computed.
        """
        if model_name not in self.records:
            return False
        train_err = self.records[model_name][0]
        valid_errs = self.records[model_name][1:]
        io.log('--> {}: train err {}, valid err {}'.format(
            model_name, train_err, valid_errs[self.args.validate_on]
        ))
        return True

    def _init_model_dir(self, model_name):
        """ Initialize directory for storing model-specific results.
        """
        model_dir = os.path.join(self.args.output_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, 0755)
        return model_dir

    def _init_resume_json(self, model_dir):
        """ Load json for resuming training if possible.
        """
        resume_json = None
        resume_fname = os.path.join(model_dir, 'resume.json')
        if os.path.exists(resume_fname):
            resume_json = io.json_load(resume_fname)
            # Check that json contains enough information
            assert 'weights' in resume_json
            assert 'lrate' in resume_json
            assert 'epoch' in resume_json
            # Make path absolute
            resume_json['weights'] = os.path.join(model_dir, resume_json['weights'])
            io.log('Resuming training: {}'.format(resume_json))
        return resume_json

    def _init_model(self, resume_json, layer_type, layer_size, num_layers,
                    bidirectional, dropout, l1_reg, l2_reg, **kwargs):
        """ Initialize a new model instance.
        """
        ext_weights_fname = self.args.ext_weights
        if resume_json is not None:
            ext_weights_fname = resume_json['weights']
        n_outs = self.buf_train.dataset().get_num_frame_classes()
        if self.args.task != 'classification':
            n_outs = 1
        return init_rnn(
            self.buf_train.dataset().get_dim(), n_outs,
            LAYERS[layer_type], layer_size, num_layers, bidirectional,
            dropout=dropout, l1_reg=l1_reg, l2_reg=l2_reg,
            ext_weights_fname=ext_weights_fname,
            non_trainable=self.args.non_trainable
        )

    def _init_lrate(self, resume_json, **kwargs):
        """ Initialize learning rate scheduler.
        """
        args = self.args
        return init_lrate(
            LRATES[args.lrate],
            args.init_lr, args.min_lr, args.depoch, args.dstart, args.dstop,
            max_epoch=args.max_epoch, resume_json=resume_json
        )

    def _init_optimizer(self, lrate, **kwargs):
        """ Initialize optimizer.
        """
        args = self.args
        return init_optimizer(
            OPTIMIZERS[args.optimizer], lrate.get_rate(),
            clip_grad=args.clip_grad
        )

    def _train(self, model, lrate, optimizer, model_name, model_dir,
               resume_json, **kwargs):
        """ Train model.
        """
        args = self.args
        loss = 'sparse_categorical_crossentropy'
        metrics = ['acc']
        if args.task != 'classification':
            loss = 'mean_squared_error'
            metrics = ['mean_squared_error']
        io.log('Loss: {}, Metrics: {}'.format(loss, metrics))
        train_errs, valid_errs = learning_rnn.train_rnn_v2(
            model, self.buf_train, self.buf_valids, lrate,
            validate_on=args.validate_on, optimizer=optimizer, save_dir=model_dir,
            pre_validate=resume_json is not None, restore=True,
            loss=loss, metrics=metrics
        )
        io.log('--> {}: train err {}, valid err {}'.format(
            model_name, train_errs[-1], valid_errs[args.validate_on][-1]
        ))
        return (train_errs, valid_errs)

    def _finalize(self, model, model_name, model_dir, results):
        """ Finalize model.
        """
        trimmed_results = self._trim_results(results)
        self._save_model(model, model_name, trimmed_results)
        self._update_internal(model_name, trimmed_results)
        self._update_records(model_name, trimmed_results)

    def _trim_results(self, results):
        """ Trim training results.
        """
        train_errs, valid_errs = results
        trimmed_results = [train_errs[-1]]
        for valid_err in valid_errs:
            trimmed_results.append(valid_err[-1])
        return trimmed_results

    def _save_model(self, model, model_name, results):
        """ Save model to disk if applicable.
        """
        args = self.args
        valid_errs = results[1:]
        if args.save_all or valid_errs[args.validate_on] < self.best_err:
            model_fname = os.path.join(args.output_dir, model_name)
            io.log('Saving final model to {}'.format(model_fname))
            io.json_save('{}.json'.format(model_fname), model.to_json())
            model.save_weights('{}.weights'.format(model_fname))

    def _update_internal(self, model_name, results):
        """ Update current best and remove previous content if applicable.
        """
        args = self.args
        valid_errs = results[1:]
        if valid_errs[args.validate_on] < self.best_err:
            io.log('Best so far! (prev: {})'.format(self.best_err))
            if not args.save_all:
                for suffix in ['json', 'weights']:
                    prev_fname = os.path.join(
                        args.output_dir, '{}.{}'.format(self.best_model, suffix)
                    )
                    if os.path.exists(prev_fname):
                        io.log('...removing {}'.format(prev_fname))
                        os.remove(prev_fname)
            self.best_model = model_name
            self.best_err = valid_errs[args.validate_on]
        else:
            io.log('Not better than prev {}'.format(self.best_err))

    def _update_records(self, model_name, results):
        """ Update records and save to disk.
        """
        self.records[model_name] = results
        save_records(self.records_fname, self.records)


    ############################
    ## Finalizing experiments ##
    ############################

    def post_execute(self):
        self._summarize()
        self._link_best_model()

    def _summarize(self):
        """ Summarize results of experiment.
        """
        io.log('==> Best model: {}, Best err: {}'.format(
            self.best_model, self.best_err
        ))

    def _link_best_model(self):
        """ Link components to the best model.
        """
        for suffix in ['json', 'weights']:
            link_name = os.path.join(
                self.args.output_dir, 'final.{}'.format(suffix)
            )
            if os.path.lexists(link_name):
                os.remove(link_name)
            os.symlink('{}.{}'.format(self.best_model, suffix), link_name)


class AdaptRNNEngine(FinetuneRNNEngine):

    #####################
    ## Argument parser ##
    #####################

    def _desc(self):
        """ Title of program to print in help text.
        """
        return 'Adapt RNN model. Supports classification and regression.'

    def _main_args(self, parser):
        """ Setup main arguments.
        """
        group = super(AdaptRNNEngine, self)._main_args(parser)
        group.add_argument('base_model_json', help='JSON of base model.')
        group.add_argument('base_model_weights', help='Weights of base model.')
        return group

    def _training_args(self, parser):
        """ Setup arguments for training.
        """
        group = super(AdaptRNNEngine, self)._training_args(parser)
        rho_help = \
            'Regularization weight based on base model to try. Accepted ' + \
            'values are in the range [0.0, 1.0). The larger the weight, ' + \
            'the more emphasis is placed on mirroring the output of the ' + \
            'base model. Regularization is enforced by minimizing the ' + \
            'KL-divergence (for classification) or MSE (for regression).'
        group.add_argument('--rho', type=float, nargs='+', default=[0.5],
                           help=rho_help)
        return group

    def _check_args(self):
        """ Perfunctory argument checks.
        """
        super(AdaptRNNEngine, self)._check_args()
        for rho in self.args.rho:
            common.CHK_RANGE(rho, 0, 1)


    #####################
    ## Initialize data ##
    #####################

    def init_data(self):
        io.log('Initializing data...')
        self._init_ivectors()
        self._init_base_data()
        self._init_base_model()
        self._init_buf_data()

    def _load_base_data(self, scp_fname, labels_fname, labels_dtype):
        """ Load a single base dataset.
        """
        args = self.args
        if is_incremental(scp_fname):
            scps = io.read_lines(scp_fname)
        else:
            scps = [scp_fname]
        data_gens = []
        for scp in scps:
            data_gens.append(DataGenerator(
                MultiLabelTemporalData.from_kaldi,
                scp=scp, alipdfs=[labels_fname], num_pdfs=[args.num_classes],
                context=args.context, padding='replicate',
                utt_feats_dict=self.ivectors, labels_dtype=labels_dtype
            ))
        return data_gens

    def _init_base_model(self):
        """ Initialize base model.
        """
        args = self.args
        io.log('Initializing base model, json: {}, weights: {}'.format(
            args.base_model_json, args.base_model_weights
        ))
        self.base_model = model_from_json(io.json_load(args.base_model_json))
        self.base_model.load_weights(args.base_model_weights)

    def _init_buf_data(self):
        """ Setup data objects for buffering.
        """
        args = self.args
        buf_cls = AugmentedBufferedUttData
        if args.use_var_utt:
            buf_cls = AugmentedBufferedVarUttData
        self.buf_train = IncrementalDataBuffer(
            self.train, buf_cls, shuffle_gens=True,
            model=self.base_model, max_frames=args.max_frames,
            nutts=args.nutts, delay=args.delay, shuffle=True
        )
        self.buf_valids = []
        for valid in self.valids:
            self.buf_valids.append(IncrementalDataBuffer(
                valid, buf_cls, shuffle_gens=False,
                model=self.base_model, max_frames=args.max_frames,
                nutts=args.nutts, delay=args.delay, shuffle=False
            ))


    #################################
    ## Preparation for experiments ##
    #################################

    def pre_execute(self):
        # Hack to fix recursion depth exceeded
        sys.setrecursionlimit(1000000)
        io.log('args: {}'.format(vars(self.args)))
        self._init_output_dir()
        self._init_records()
        self._init_base_err()
        self._init_best_err()

    def _init_base_err(self):
        """ Compute validation results using base model.
        """
        args = self.args
        if 'base' in self.records:
            valid_errs = self.records['base'][1:]
            self.base_err = valid_errs[args.validate_on]
            return
        metrics = ['acc']
        if args.task != 'classification':
            metrics = ['mean_squared_error']
        io.log('Reporting base model results, metrics = {}'.format(metrics))
        # Convert to multi-task to conform with data provider
        mt_model = Model(
            input=self.base_model.input, output=[self.base_model.output] * 2
        )
        train_err = -1  # Don't compute error on training set, too expensive
        valid_errs = []
        for buf_valid in self.buf_valids:
            valid_errs.append(learning_mtrnn.validate_mtrnn(
                mt_model, buf_valid, metrics=metrics
            )[0][0])
        io.log('--> base: train err {}, valid err {}'.format(
            train_err, valid_errs[args.validate_on]
        ))
        # Update records and clean up
        self._update_records('base', [train_err] + valid_errs)
        self.base_err = valid_errs[args.validate_on]
        del mt_model


    ##########################
    ## Main experiment code ##
    ##########################

    def params_iter(self):
        args = self.args
        params_title = [
            'layer_type', 'layer_size', 'num_layers',
            'bidirectional', 'dropout', 'l1_reg', 'l2_reg', 'rho'
        ]
        params_iter = itertools.product(
            args.layer_type, args.layer_size, args.num_layers,
            args.bidirectional, args.dropout, args.l1_reg, args.l2_reg, args.rho
        )
        return (params_title, params_iter)

    def _model_name(self, rho, **kwargs):
        """ Return a string that identifies this model combination.
        """
        model_name = super(AdaptRNNEngine, self)._model_name(**kwargs)
        return '{}+rho_{}'.format(model_name, rho)

    def _init_model(self, resume_json, layer_type, layer_size, num_layers,
                    bidirectional, dropout, l1_reg, l2_reg, **kwargs):
        """ Initialize a new model instance.
        """
        ext_weights_fname = self.args.ext_weights
        if resume_json is not None:
            ext_weights_fname = resume_json['weights']
        n_outs = self.buf_train.dataset().get_num_frame_classes()
        if self.args.task != 'classification':
            n_outs = 1
        model = init_rnn(
            self.buf_train.dataset().get_dim(), n_outs,
            LAYERS[layer_type], layer_size, num_layers, bidirectional,
            dropout=dropout, l1_reg=l1_reg, l2_reg=l2_reg,
            ext_weights_fname=ext_weights_fname,
            non_trainable=self.args.non_trainable, input=self.base_model.input
        )
        # Convert to multi-task to conform with data provider
        return Model(input=model.input, output=[model.output] * 2)

    def _train(self, model, lrate, optimizer, model_name, model_dir,
               resume_json, rho, **kwargs):
        """ Train model.
        """
        args = self.args
        loss = ['sparse_categorical_crossentropy', 'kullback_leibler_divergence']
        metrics = ['acc']
        validate_metrics = { 0: ['acc'], 1: ['mean_squared_error'] }
        if args.task != 'classification':
            loss = ['mean_squared_error', 'mean_squared_error']
            metrics = ['mean_squared_error']
            validate_metrics = ['mean_squared_error']
        task_weights = [1.0 - rho, rho]
        io.log('Loss: {}, Metrics: {}, Task Weights: {}'.format(
            loss, validate_metrics, task_weights
        ))
        train_errs, valid_errs = learning_mtrnn.train_mtrnn_v2(
            model, self.buf_train, self.buf_valids, lrate,
            validate_on=args.validate_on, optimizer=optimizer, save_dir=model_dir,
            pre_validate=resume_json is not None, restore=True,
            loss=loss, metrics=metrics, validate_metrics=validate_metrics,
            ntasks=2, primary_task=0, task_weights=task_weights
        )
        io.log('--> {}: train err {}, valid err {}'.format(
            model_name, train_errs[-1], valid_errs[args.validate_on][-1]
        ))
        return (train_errs, valid_errs)

    def _save_model(self, model, model_name, results):
        """ Save model to disk if applicable.
        """
        args = self.args
        valid_errs = results[1:]
        if args.save_all or valid_errs[args.validate_on] < self.best_err:
            model_fname = os.path.join(args.output_dir, model_name)
            io.log('Saving final model to {}'.format(model_fname))
            # Remove auxiliary output prior to saving
            st_model = Model(input=model.input, output=model.output[0])
            io.json_save('{}.json'.format(model_fname), st_model.to_json())
            st_model.save_weights('{}.weights'.format(model_fname))
            del st_model


    ############################
    ## Finalizing experiments ##
    ############################

    def _summarize(self):
        """ Summarize results of experiment.
        """
        io.log('==> Best model: {}, Best err: {} (diff vs. base {})'.format(
            self.best_model, self.best_err, self.best_err - self.base_err
        ))

