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
from collections import OrderedDict

######################################
## Useful general-purpose functions ##
#####################################

def make_index(keys, values, ordered=False):
    CHK_EQ(len(keys), len(values))
    index = OrderedDict() if ordered else {}
    for key, value in zip(keys, values):
        index[key] = value
    return index

def make_position_index(lst, ordered=False):
    """ Create a mapping from list item to its index in the list """
    index = OrderedDict() if ordered else {}
    for i, item in zip(range(len(lst)), lst):
        index[item] = i
    return index

def make_reverse_index(mapping, ordered=False):
    """ Create a reverse index from a dictionary, i.e. mapping from value
    to a list of keys.
    """
    reverse_index = OrderedDict() if ordered else {}
    for key, value in mapping.items():
        if value not in reverse_index:
            reverse_index[value] = []
        reverse_index[value].append(key)
    return reverse_index

def merge_dicts(*dict_args):
    """ Merge an arbitrary number of dicts into a new dict using shallow copy.
    Precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def rm_ext(fname):
    """ Remove extension from file name.
    """
    idx = fname.rfind(".")
    return fname if idx == -1 else fname[:idx]

def init_argparse(desc):
    """ Initialize argument parser.
    """
    import argparse
    parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    return parser

def ms_to_s(ms):
    """ Convert ms to s.
    """
    return ms / 1000.0

def s_to_ms(s):
    """ Convert s to ms.
    """
    return s * 1000.0

def find_all(enumerable, to_find):
    """ Return a list of indices containing occurrences of `to_find` in
    `enumerable`. If no occurrence, return an empty list.
    """
    return [index for index, item in enumerate(enumerable) if item == to_find]

def is_ascii(text):
    """ Return True if `text` is in valid ASCII, False otherwise.
    """
    try:
        text.decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

#######################################################
## Assertions with user-friendly diagnostic messages ##
#######################################################

def CHK_GT(x, y):
    assert x > y, "{} is not greater than {}".format(x, y)

def CHK_GE(x, y):
    assert x >= y, "{} is not greater than or equal to {}".format(x, y)

def CHK_LT(x, y):
    assert x < y, "{} is not smaller than {}".format(x, y)

def CHK_LE(x, y):
    assert x <= y, "{} is not smaller than or equal to {}".format(x, y)

def CHK_EQ(x, y):
    assert x == y, "{} is not equal to {}".format(x, y)

def CHK_NEQ(x, y):
    assert x != y, "{} is equal to {}".format(x, y)

def CHK_RANGE(x, lo, hi, inclusive=False):
    assert x >= lo and (x < hi or (inclusive and x <= hi)), \
            "Condition not met: {} <= {} {} {}".format(
                lo, x, "<=" if inclusive else "<", hi
            )

def CHK_DIM(ary, dim, print_content=False):
    CHK_DIMS(ary, [dim], print_content=print_content)

def CHK_DIMS(ary, dims, print_content=False):
    assert len(ary.shape) in dims, \
            "Invalid shape: {} ({}). Supported dims: {}".format(
                    ary.shape, ary if print_content else "omitted", dims
            )

def CHK_LEN(lst, length, print_content=False):
    CHK_LENS(lst, [length], print_content=print_content)

def CHK_LENS(lst, lengths, print_content=False):
    assert len(lst) in lengths, \
            "Invalid length: {} ({}). Supported lengths: {}".format(
                    len(lst), lst if print_content else "omitted", lengths
            )

def CHK_VAL(x, val):
    CHK_VALS(x, [val])

def CHK_VALS(x, vals):
    assert x in vals, "Invalid value: {}. Accepted values: {}".format(x, vals)

##############################
## Generally useful classes ##
##############################

class TimeTag(object):
    """
    A general-purpose tag that has a name, value, start, and end time. The exact
    semantics of `name`, `value`, `start`, and `end` are defined by the caller.
    """
    def __init__(self, name, start=0, end=0, value=0.0):
        self.name = name
        self.start = start
        self.end = end
        self.value = value

    def get_value(self):
        return self.value

    def get_duration(self):
        return self.end - self.start

    def offset(self, n):
        self.start = self.start + n
        self.end = self.end + n

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return '{} ({}) {}->{}'.format(
            self.name, self.get_value(), self.start, self.end
        )

    def __repr__(self):
        return self.__str__()


class ExtendedTimeTag(TimeTag):
    def __init__(self, name):
        super(ExtendedTimeTag, self).__init__(name, 0, 0)
        self.sub_tags = []

    def add_tags(self, sub_tags):
        for tag in sub_tags:
            self.add_tag(tag)

    def add_tag(self, sub_tag):
        self.sub_tags.append(sub_tag)
        if len(self.sub_tags) == 1:
            self.start = sub_tag.start
        self.end = sub_tag.end

    def offset(self, n):
        super(ExtendedTimeTag, self).offset(n)
        for tag in self.sub_tags:
            tag.offset(n)

    def get_value(self):
        sum = 0.0
        for tag in self.sub_tags:
            sum += tag.get_value()
        return sum

