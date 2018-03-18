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

import os
import numpy as np

import chaipy.common as common
import chaipy.io as io
import chaipy.learning.mtrnn as learning_mtrnn
from chaipy.data import is_incremental
from chaipy.data.temporal import MultiLabelTemporalData
from chaipy.data.incremental import DataGenerator

from rnn import LAYERS, OPTIMIZERS, LRATES, TASKS, init_rnn, \
                FinetuneRNNEngine, AdaptRNNEngine

# keras imports
from keras.models import Model


def get_task_weights(task_weights, rho):
    weights = []
    weights.extend(task_weights)
    for i in range(len(task_weights)):
        scaled_rho = rho * weights[i]
        weights.append(scaled_rho)
        weights[i] -= scaled_rho
    return weights


class FinetuneMTRNNEngine(FinetuneRNNEngine):

    #####################
    ## Argument parser ##
    #####################

    def _desc(self):
        """ Title of program to print in help text.
        """
        return 'Finetune multi-task RNN model. Supports classification, ' + \
               'regression, or a mix of both.'

    def _register_custom_types(self, parser):
        """ Register custom types for arguments.
        """
        parser = super(FinetuneMTRNNEngine, self)._register_custom_types(parser)
        parser.register('type', 'int', lambda x: None if x == '-1' else int(x))
        return parser

    def _main_args(self, parser):
        """ Setup main arguments.
        """
        group = parser.add_argument_group(title='Main Arguments')
        group.add_argument('train_scp', help='scp of training data.')
        group.add_argument('train_labels', nargs='+',
                           help='Labels of training data (must specify ' + \
                                'at least one, can have multiple).')
        group.add_argument('output_dir', help='Output directory to save results.')
        return group

    def _validation_args(self, parser):
        """ Setup arguments for validation sets.
        """
        group = parser.add_argument_group(title='Validation Setup')
        group.add_argument('--valid-scp', nargs='+', default=[],
                           help='scp of validation data (must specify ' + \
                                'at least one, can have multiple).')
        valid_labels_help = \
            'Labels of validation data. Specify labels in the same order ' + \
            'as `valid-scp`. For example, given `--valid-scp scp1 scp2` ' + \
            'and `--valid-labels lbl1 lbl2 lbl3 lbl4`, scp1 will be ' + \
            'assigned lbl1 and lbl2; scp2 will be assigned lbl3 and lbl4.'
        group.add_argument('--valid-labels', nargs='+', default=[],
                           help=valid_labels_help)
        group.add_argument('--validate-on', type=int, default=0,
                           help='Which validation set to validate on.')
        return group

    def _data_args(self, parser):
        """ Setup arguments for data provider.
        """
        group = parser.add_argument_group(title='Data Provider Configuration')
        task_help = \
            'Target tasks, one for each label ({}). '.format(list(TASKS)) + \
            'If only one is specified, apply the same task to all labels.'
        group.add_argument('--task', nargs='+', default=['classification'],
                           help=task_help)
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
        num_classes_help = \
            'Number of classes, one for each task. If not set, use max ' + \
            'value in `train_labels`. For tasks with no classes ' + \
            '(e.g., regression), use -1, which will be converted to None.'
        group.add_argument('--num-classes', nargs='+', type='int', default=None,
                           help=num_classes_help)
        return group

    def _training_args(self, parser):
        """ Setup arguments for training.
        """
        group = super(FinetuneMTRNNEngine, self)._training_args(parser)
        group.add_argument('--task-weights', nargs='+', type=float, default=None,
                           help='Task weights; if not set, use 1.0 everywhere.')
        group.add_argument('--primary-task', type=int, default=0,
                           help='Which task to validate on.')
        return group

    def _check_args(self):
        """ Perfunctory argument checks and modification.
        """
        args = self.args
        # Check scp and labels
        common.CHK_GE(len(args.valid_scp), 1)
        common.CHK_EQ(len(args.valid_labels) % len(args.valid_scp), 0)
        labels_per_valid_scp = len(args.valid_labels) / len(args.valid_scp)
        common.CHK_EQ(len(args.train_labels), labels_per_valid_scp)
        # Check task
        if len(args.task) == 1:
            args.task = args.task * len(args.train_labels)
        common.CHK_EQ(len(args.task), len(args.train_labels))
        for task in args.task:
            common.CHK_VALS(task, TASKS)
        if args.num_classes is not None:
            common.CHK_EQ(len(args.task), len(args.num_classes))
        if args.task_weights is None:
            args.task_weights = [1.0] * len(args.task)
        common.CHK_EQ(len(args.task), len(args.task_weights))
        # Check others
        for layer_type in args.layer_type:
            common.CHK_VALS(layer_type, LAYERS.keys())
        common.CHK_VALS(args.optimizer, OPTIMIZERS.keys())
        common.CHK_VALS(args.lrate, LRATES.keys())


    #####################
    ## Initialize data ##
    #####################

    def _init_base_data(self):
        """ Setup base data objects.
        """
        args = self.args
        labels_dtype = map(
            lambda x: np.int32 if x == 'classification' else np.float32,
            args.task
        )
        self.train = self._load_base_data(
            args.train_scp, args.train_labels, labels_dtype
        )
        self.valids = []
        lbls = args.valid_labels
        x = len(lbls) / len(args.valid_scp)
        split_valid_labels = [lbls[i:i+x] for i in range(0, len(lbls), x)]
        for valid_scp, valid_labels in zip(args.valid_scp, split_valid_labels):
            self.valids.append(self._load_base_data(
                valid_scp, valid_labels, labels_dtype
            ))

    def _load_base_data(self, scp_fname, labels_fnames, labels_dtype):
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
                scp=scp, alipdfs=labels_fnames, num_pdfs=args.num_classes,
                context=args.context, padding='replicate',
                utt_feats_dict=self.ivectors, labels_dtype=labels_dtype
            ))
        return data_gens


    ##########################
    ## Main experiment code ##
    ##########################

    def _init_model(self, resume_json, layer_type, layer_size, num_layers,
                    bidirectional, dropout, l1_reg, l2_reg, **kwargs):
        """ Initialize a new model instance.
        """
        ext_weights_fname = self.args.ext_weights
        if resume_json is not None:
            ext_weights_fname = resume_json['weights']
        n_outs = []
        for i in range(len(self.args.task)):
            if self.args.task[i] == 'classification':
                n_outs.append(self.buf_train.dataset().get_num_frame_classes()[i])
            else:
                n_outs.append(1)
        return init_rnn(
            self.buf_train.dataset().get_dim(), n_outs,
            LAYERS[layer_type], layer_size, num_layers, bidirectional,
            dropout=dropout, l1_reg=l1_reg, l2_reg=l2_reg,
            ext_weights_fname=ext_weights_fname,
            non_trainable=self.args.non_trainable
        )

    def _train(self, model, lrate, optimizer, model_name, model_dir,
               resume_json, **kwargs):
        """ Train model.
        """
        args = self.args
        loss = []
        validate_metrics = {}
        for i in range(len(args.task)):
            if args.task[i] == 'classification':
                loss.append('sparse_categorical_crossentropy')
                validate_metrics[i] = ['acc']
            else:
                loss.append('mean_squared_error')
                validate_metrics[i] = ['mean_squared_error']
        metrics = ['acc']
        if args.task[args.primary_task] != 'classification':
            metrics = ['mean_squared_error']
        io.log('Loss: {}, Metrics: {}, Primary Task: {}, Task Weights: {}'.format(
            loss, validate_metrics, args.primary_task, args.task_weights
        ))
        train_errs, valid_errs = learning_mtrnn.train_mtrnn_v2(
            model, self.buf_train, self.buf_valids, lrate,
            validate_on=args.validate_on, optimizer=optimizer, save_dir=model_dir,
            pre_validate=resume_json is not None, restore=True,
            loss=loss, metrics=metrics, validate_metrics=validate_metrics,
            ntasks=len(args.task), primary_task=args.primary_task,
            task_weights=args.task_weights
        )
        io.log('--> {}: train err {}, valid err {}'.format(
            model_name, train_errs[-1], valid_errs[args.validate_on][-1]
        ))
        return (train_errs, valid_errs)


class AdaptMTRNNEngine(FinetuneMTRNNEngine, AdaptRNNEngine):

    #####################
    ## Argument parser ##
    #####################

    def _desc(self):
        """ Title of program to print in help text.
        """
        return 'Adapt multi-task RNN model. Supports classification, ' + \
               'regression, or a mix of both.'

    def _main_args(self, parser):
        """ Setup main arguments.
        """
        group = super(AdaptMTRNNEngine, self)._main_args(parser)
        group.add_argument('base_model_json', help='JSON of base model.')
        group.add_argument('base_model_weights', help='Weights of base model.')
        return group

    def _check_args(self):
        """ Perfunctory argument checks and modification.
        """
        super(AdaptMTRNNEngine, self)._check_args()
        for rho in self.args.rho:
            common.CHK_RANGE(rho, 0, 1)


    #####################
    ## Initialize data ##
    #####################

    def init_data(self):
        AdaptRNNEngine.init_data(self)

    def _init_base_model(self):
        AdaptRNNEngine._init_base_model(self)

    def _init_buf_data(self):
        AdaptRNNEngine._init_buf_data(self)


    #################################
    ## Preparation for experiments ##
    #################################

    def pre_execute(self):
        AdaptRNNEngine.pre_execute(self)

    def _init_base_err(self):
        """ Compute validation results using base model.
        """
        args = self.args
        if 'base' in self.records:
            valid_errs = self.records['base'][1:]
            self.base_err = valid_errs[args.validate_on]
            return
        metrics = ['acc']
        if args.task[args.primary_task] != 'classification':
            metrics = ['mean_squared_error']
        io.log('Reporting base model results, metrics = {}'.format(metrics))
        train_err = -1  # Don't compute error on training set, too expensive
        valid_errs = []
        for buf_valid in self.buf_valids:
            valid_errs.append(learning_mtrnn.validate_mtrnn(
                self.base_model, buf_valid, metrics=metrics
            )[args.primary_task][0])
        io.log('--> base: train err {}, valid err {}'.format(
            train_err, valid_errs[args.validate_on]
        ))
        # Update records and clean up
        self._update_records('base', [train_err] + valid_errs)
        self.base_err = valid_errs[args.validate_on]


    ##########################
    ## Main experiment code ##
    ##########################

    def params_iter(self):
        return AdaptRNNEngine.params_iter(self)

    def _model_name(self, **kwargs):
        return AdaptRNNEngine._model_name(self, **kwargs)

    def _init_model(self, **kwargs):
        """ Initialize a new model instance.
        """
        model = FinetuneMTRNNEngine._init_model(self, **kwargs)
        # Add auxiliary tasks
        output = model.output
        if type(model.output) != list:
            output = [model.output]
        return Model(input=model.input, output=output * 2)

    def _train(self, model, lrate, optimizer, model_name, model_dir,
               resume_json, rho, **kwargs):
        """ Train model.
        """
        args = self.args
        loss = []
        aux_loss = []
        validate_metrics = {}
        for i in range(len(args.task)):
            if args.task[i] == 'classification':
                loss.append('sparse_categorical_crossentropy')
                aux_loss.append('kullback_leibler_divergence')
                validate_metrics[i] = ['acc']
            else:
                loss.append('mean_squared_error')
                aux_loss.append('mean_squared_error')
                validate_metrics[i] = ['mean_squared_error']
            # KLD metric
            validate_metrics[i + len(args.task)] = ['mean_squared_error']
        metrics = ['acc']
        if args.task[args.primary_task] != 'classification':
            metrics = ['mean_squared_error']
        task_weights = get_task_weights(args.task_weights, rho)
        io.log('Loss: {}, Metrics: {}, Primary Task: {}, Task Weights: {}'.format(
            loss + aux_loss, validate_metrics, args.primary_task, task_weights
        ))
        train_errs, valid_errs = learning_mtrnn.train_mtrnn_v2(
            model, self.buf_train, self.buf_valids, lrate,
            validate_on=args.validate_on, optimizer=optimizer, save_dir=model_dir,
            pre_validate=resume_json is not None, restore=True,
            loss=loss + aux_loss, metrics=metrics,
            validate_metrics=validate_metrics, ntasks=len(args.task) * 2,
            primary_task=args.primary_task, task_weights=task_weights
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
            trimmed_model = Model(
                input=model.input, output=model.output[0:len(args.task)]
            )
            io.json_save('{}.json'.format(model_fname), trimmed_model.to_json())
            trimmed_model.save_weights('{}.weights'.format(model_fname))
            del trimmed_model


    ############################
    ## Finalizing experiments ##
    ############################

    def _summarize(self):
        AdaptRNNEngine._summarize(self)

