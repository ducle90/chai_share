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

import chaipy.common as common
import chaipy.io as io

from temporal import BufferedData


class DataGenerator(object):
    """
    This wrapper class simply defines how the dataset is generated. The actual
    creation of the dataset is delayed until it's needed.
    """
    def __init__(self, generator_fn, **kwargs):
        """
        :type generator_fn: function
        :param generator_fn: Function used to generate the dataset

        :type kwargs: dict
        :param kwargs: Keyword arguments to pass to `generator_fn`
        """
        self._generator_fn = generator_fn
        self._kwargs = kwargs

    def generate(self):
        return self._generator_fn(**self._kwargs)


class IncrementalDataBuffer(BufferedData):
    """
    This class allows buffering on a set of `DataGenerator` objects instead of
    a single dataset. When data from one generator is exhausted, the code swaps
    to a new generator, until all generators have been processed. This way,
    only one dataset is loaded at any given time, avoiding the need to load
    all datasets into memory at once.
    """
    def __init__(self, data_gens, buf_cls, shuffle_gens=True, **kwargs):
        """
        :type buf_cls: class
        :param buf_cls: Class to use for buffered dataset

        :type data_gens: list of chaipy.data.incremental.DataGenerator
        :param data_gens: For generating individual datasets

        :type shuffle_gens: bool
        :param shuffle_gens: Shuffle generator list after each iteration

        :type kwargs: dict
        :param kwargs: Keyword arguments used to initialize `buf_cls`
        """
        common.CHK_EQ(type(data_gens), list)
        common.CHK_GE(len(data_gens), 1)
        self.__buf_dataset = None
        self._data_gens = data_gens
        self._buf_cls = buf_cls
        self._shuffle_gens = shuffle_gens
        self._kwargs = kwargs
        self.reset()

    def reset(self):
        self.current_index = 0
        if self._shuffle_gens:
            numpy.random.shuffle(self._data_gens)
        # Avoid reloading dataset if there's only one data generator
        if self.__buf_dataset is None or len(self._data_gens) != 1:
            self.__buf_dataset = self._load_dataset()

    def _load_dataset(self):
        dataset = self._data_gens[self.current_index].generate()
        return self._buf_cls(dataset=dataset, **self._kwargs)

    def dataset(self):
        return self.__buf_dataset.dataset()

    def read_next_chunk(self):
        if self.current_index >= len(self._data_gens):
            self.reset()
            return None
        else:
            X = self.__buf_dataset.read_next_chunk()
            if X is not None:
                return X
            # Load next data generator
            self.current_index += 1
            if self.current_index < len(self._data_gens):
                self.__buf_dataset = self._load_dataset()
            return self.read_next_chunk()

    def get_delay(self):
        return self.__buf_dataset.get_delay()

    def get_progress(self):
        sub_progress = self.__buf_dataset.get_progress()
        progress = '{}/{}'.format(self.current_index + 1, len(self._data_gens))
        return '{}:{}'.format(sub_progress, progress)

