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
from itertools import chain, combinations

import chaipy.common as common
import chaipy.io as io


def __report(i, report_interval):
    if report_interval is not None and i % report_interval == 0:
        io.log(i)


def fill_vals(vals, xs, fn, report_interval=None):
    for i, x in zip(range(len(xs)), xs):
        __report(i + 1, report_interval)
        vals[i] = fn(x)


def fill_ary(ary, xs, fn, report_interval=None):
    index = 0
    for i, x in zip(range(len(xs)), xs):
        __report(i + 1, report_interval)
        tmp_ary = fn(x)
        ary[index:index+len(tmp_ary)] = tmp_ary
        index += len(tmp_ary)


def accum_val(xs, fn, start=0, report_interval=None):
    for i, x in zip(range(len(xs)), xs):
        __report(i + 1, report_interval)
        start += fn(x)
    return start


def parse_num_bytes(size_str):
    """ Return the number of bytes that corresponds to a human-readable
    size string. The following formats are currently accepted:
        * <num>m or <num>M - e.g. '600m' or '600M' means 600 megabytes
        * <num>g or <num>G - e.g. '1g' or '1G' means 1 gigabyte
    """
    if size_str[-1] in ['m', 'M']:
        return (int) ((1024 ** 2) * float(size_str[0:-1]))
    elif size_str[-1] in ['g', 'G']:
        return (int) ((1024 ** 3) * float(size_str[0:-1]))
    else:
        raise ValueError("Invalid size string: {}".format(size_str))


def get_num_items(size_str, dtype):
    """ Return the number of items of type dtype that can be fit in a memory
    chunk of certain size. For example, get_num_items('600m', numpy.float32)
    will return how many float32 numbers can be fit inside a 600 MB chunk.
    """
    return parse_num_bytes(size_str) / numpy.dtype(dtype).itemsize


def normalize(ary, dtype=numpy.float32):
    """ Return an array of the same shape normalized at the row level, i.e.
    each row forms a probability distribution.
    """
    ary = numpy.asarray(ary, dtype=dtype)
    if len(ary.shape) == 1:
        return ary / numpy.sum(ary)
    else:
        norm_shape = []
        for i in range(len(ary.shape)):
            dim = ary.shape[i]
            if i == len(ary.shape) - 1:
                dim = 1
            norm_shape.append(dim)
        norm_shape = tuple(norm_shape)
        return ary / numpy.sum(ary, axis=-1).reshape(norm_shape)


def partition(ary, fraction):
    """ Partition the array into N parts, where the number of elements in
    each part is roughly fraction times the length of the original array.
    """
    common.CHK_RANGE(fraction, 0, 1, inclusive=True)
    step = max(1, int(round(len(ary) * fraction)))
    split_indices = numpy.arange(step, len(ary), step)
    return numpy.split(ary, split_indices)


def powerset(iterable, minlen=0):
    s = list(iterable)
    subset_lens = range(minlen, len(s) + 1)
    return chain.from_iterable(combinations(s, r) for r in subset_lens)


def is_incremental(scp):
    """ Return True if the scp indicates an incremental buffer, False otherwise.
    """
    with open(scp, 'r') as f:
        line = f.readline().strip()
    return len(line.split()) == 1

