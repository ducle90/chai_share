# Copyright 2013    Yajie Miao    Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, do_maxout = False, pool_size = 1):
        """ Class for hidden layer """
        self.input = input
        self.n_in = n_in
        self.n_out = n_out

        self.activation = activation

        self.type = 'fc'
        
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if self.activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        lin_output = T.dot(input, self.W) + self.b
        if do_maxout == True:
            self.last_start = n_out - pool_size
            self.tmp_output = lin_output[:,0:self.last_start+1:pool_size]
            for i in range(1, pool_size):
                cur = lin_output[:,i:self.last_start+i+1:pool_size]
                self.tmp_output = T.maximum(cur, self.tmp_output)
            self.output = self.activation(self.tmp_output)
        else:
            self.output = (lin_output if self.activation is None
                           else self.activation(lin_output))

        # parameters of the model
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

def _dropout_from_layer(theano_rng, hid_out, p):
    """ p is the factor for dropping a unit """
    # p=1-p because 1's indicate keep and p is prob of dropping
    return theano_rng.binomial(n=1, p=1-p, size=hid_out.shape,
                               dtype=theano.config.floatX) * hid_out

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 W=None, b=None, activation=T.tanh, do_maxout = False, pool_size = 1, dropout_factor=0.5):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, do_maxout = do_maxout, pool_size = pool_size)

        self.theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.dropout_output = _dropout_from_layer(theano_rng = self.theano_rng,
                                                  hid_out = self.output, p=dropout_factor)

class FactoredHiddenLayer(object):
    """ Use this when you want a portion of the input data to be modeled
    by a separate network. For example, the i-vector can be modeled separately
    by a small hidden layer, and the output will be added to the layer output.
    """
    def __init__(self, rng, input, n_in, n_in_main, n_in_side, n_out,
                 side_layers, W=None, b=None, activation=T.tanh):
        self.input = input
        self.n_in = n_in
        self.n_in_main = n_in_main
        self.n_in_side = n_in_side
        self.n_out = n_out
        self.side_layers = side_layers

        self.activation = activation
        self.type = 'factored'
        
        # Main weights
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in_main + n_out)),
                    high=numpy.sqrt(6. / (n_in_main + n_out)),
                    size=(n_in_main, n_out)), dtype=theano.config.floatX)
            if self.activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        self.delta_W = theano.shared(value = numpy.zeros((n_in_main, n_out),
                                     dtype=theano.config.floatX), name='delta_W')
        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        # Secondary weights
        self.side_Ws, self.side_delta_Ws = [], []
        self.side_bs, self.side_delta_bs = [], []
        for i in range(len(side_layers) + 1):
            fan_in = side_layers[i-1] if i > 0 else n_in_side
            fan_out = side_layers[i] if i < len(side_layers) else n_out
            side_W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (fan_in + fan_out)),
                    high=numpy.sqrt(6. / (fan_in + fan_out)),
                    size=(fan_in, fan_out)), dtype=theano.config.floatX)
            if self.activation == theano.tensor.nnet.sigmoid:
                side_W_values *= 4
            self.side_Ws.append(theano.shared(
                value=side_W_values, name='side_W_{}'.format(i), borrow=True
            ))
            self.side_delta_Ws.append(theano.shared(
                value=numpy.zeros((fan_in, fan_out), dtype=theano.config.floatX),
                name='delta_W_{}'.format(i)
            ))
            # No bias needed for outermost weight
            if i < len(side_layers):
                side_b_values = numpy.zeros((fan_out,), dtype=theano.config.floatX)
                self.side_bs.append(theano.shared(
                    value=side_b_values, name='side_b_{}'.format(i), borrow=True
                ))
                self.side_delta_bs.append(theano.shared(
                    value=numpy.zeros_like(self.side_bs[-1].get_value(borrow=True),
                                           dtype=theano.config.floatX),
                    name='delta_b_{}'.format(i)
                ))
        
        side_outputs = []
        for i in range(len(side_layers)):
            if i == 0:
                lin_side_output = \
                        T.dot(input[:,n_in-n_in_side:], self.side_Ws[i]) + \
                        self.side_bs[i]
            else:
                lin_side_output = \
                        T.dot(side_outputs[i-1], self.side_Ws[i]) + \
                        self.side_bs[i]
            side_outputs.append(lin_side_output if self.activation is None \
                                else self.activation(lin_side_output))

        lin_output = T.dot(input[:,:n_in_main], self.W) + self.b + \
                T.dot(side_outputs[-1], self.side_Ws[-1])
        self.output = lin_output if self.activation is None \
                      else self.activation(lin_output)

        # Parameters of the model
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]
        self.params.extend(self.side_Ws)
        self.params.extend(self.side_bs)
        self.delta_params.extend(self.side_delta_Ws)
        self.delta_params.extend(self.side_delta_bs)

class WeightedFactoredHiddenLayer(object):
    """ Use this when you want the input data to be modeled by a weighted sum
    of several separate layers. The last features of the input are weights.
    """
    def __init__(self, rng, input, n_in, n_weights, n_out,
                 W=None, b=None, activation=T.tanh):
        self.input = input
        self.n_in = n_in
        self.n_weights = n_weights
        n_in_main = n_in - n_weights
        self.n_out = n_out

        self.activation = activation
        self.type = 'wfactored'

        self.W = []
        for i in range(n_weights):
            if W is None:
                W_values = numpy.asarray(rng.uniform(
                        low=-numpy.sqrt(6. / (n_in_main + n_out)),
                        high=numpy.sqrt(6. / (n_in_main + n_out)),
                        size=(n_in_main, n_out)), dtype=theano.config.floatX)
                if self.activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4
                W = theano.shared(value=W_values, name='W_{}'.format(i), borrow=True)
            else:
                assert "Can't have shared weights in factored layer!"
            self.W.append(W)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.b = b

        self.delta_W = []
        for i in range(n_weights):
            self.delta_W.append(theano.shared(
                value=numpy.zeros((n_in_main, n_out), dtype=theano.config.floatX),
                name='delta_W_{}'.format(i)
            ))
        self.delta_b = theano.shared(
            value=numpy.zeros_like(self.b.get_value(borrow=True),
                                   dtype=theano.config.floatX), name='delta_b')

        lin_output = self.b
        for i in range(n_weights):
            start = n_in_main + i
            lin_output += self.input[:,start:start+1].repeat(n_out, axis=1) * \
                    T.dot(self.input[:,:n_in_main], self.W[i])
        self.output = lin_output if self.activation is None \
                      else self.activation(lin_output)

        # Parameters of the model
        self.params = self.W + [self.b]
        self.delta_params = self.delta_W + [self.delta_b]
