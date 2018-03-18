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

import chaipy.common as common
import sys
import json
import pickle
import numpy
from collections import OrderedDict
from datetime import datetime


def dict_read(fname, ordered=False, fn=None, skip_empty=False, lst=False):
    """ Read a dictionary from file, assuming tokens are separated by
    whitespace. If there's more than one value token, each key will map to
    a list of values instead of a single value.

    If ordered=True, keys will be sorted in the order that they are inserted.
    If fn is set, it will be used to transform each value.
    If skip_empty=True, empty lines will be skipped.
    If lst=True, each key will always map to a list of values.
    """
    mappings = OrderedDict() if ordered else {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            ary = line.strip().split()
            if len(ary) == 1:
                if skip_empty:
                    continue
                mappings[ary[0]] = [] if lst else None
            elif len(ary) == 2 and not lst:
                mappings[ary[0]] = ary[1] if fn is None else fn(ary[1])
            else:
                mappings[ary[0]] = ary[1:] if fn is None else \
                        [fn(v) for v in ary[1:]]
    return mappings


def dict_write(fname, key2val, sep=' ', fn=None):
    """ Write content of a dictionary to file.

    :type sep: str
    :param sep: (optional) Separator between key and value

    :type fn: function
    :param fn: (optional) Function that takes a dictionary value and returns
        a string to print out. If not specified, use built-in representation.
    """
    lines = []
    for key in key2val:
        val_str = key2val[key]
        if fn is not None:
            val_str = fn(val_str)
        lines.append('{}{}{}'.format(key, sep, val_str))
    write_lines(fname, lines)


def lexicon_read(fname, ordered=False):
    """ Read a lexicon from file, assuming tokens are separated by whitespace,
    where the first token is the word name and the remaining tokens are phones.
    This function returns a map where keys are words and each value is a list
    of pronunciations. Each pronunciation is a list of phones.

    If ordered=True, keys will be sorted in the order that they are inserted.
    """
    lexicon = OrderedDict() if ordered else {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            ary = line.strip().split()
            if len(ary) == 1:
                log('WARNING - empty line: {}'.format(line))
            if ary[0] not in lexicon:
                lexicon[ary[0]] = []
            lexicon[ary[0]].append(ary[1:])
    return lexicon


def json_save(fname, json_obj, sort_keys=False, indent=4):
    """ Save a JSON object, i.e. simple dictionary to file """
    json.dump(json_obj, open(fname, 'w'), sort_keys=sort_keys, indent=indent)


def json_load(fname, ordered=False):
    """ Load a JSON object from file """
    object_pairs_hook = OrderedDict if ordered else dict
    return json.load(open(fname, 'r'), object_pairs_hook=object_pairs_hook)


def pickle_save(fname, obj, protocol=pickle.HIGHEST_PROTOCOL):
    """ Simple method for saving a single object as a pickled file """
    with open(fname, 'wb') as output:
        pickle.dump(obj, output, protocol)


def pickle_load(fname):
    """ Simple method for loading a single object from a pickled file """
    with open(fname, 'rb') as input:
        obj = pickle.load(input)
    return obj


def log(msg, stream=sys.stderr):
    print >> stream, '[{}] {}'.format(datetime.now(), msg)


def read_lines(fname, strip=True):
    lines = []
    with open(fname, 'r') as f:
        for line in f:
            lines.append(line.strip() if strip else line)
    return lines


def write_lines(fname, lines, add_newline=True):
    """ Write `lines` to `fname`. """
    fw = open(fname, 'w')
    for line in lines:
        fw.write(line)
        if add_newline:
            fw.write('\n')
    fw.close()

