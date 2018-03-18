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
from sklearn.mixture import GMM
from chaipy.io import log

def train_gmms(group2array, n_components=1, cov_type='diag', ordered=True):
    """ Train a GMM for each group. Return group to GMM mapping. """
    gmms = OrderedDict() if ordered else {}
    for group in group2array:
        log('Fitting GMM for {}'.format(group))
        gmms[group] = GMM(n_components=n_components, covariance_type=cov_type)
        gmms[group].fit(group2array[group])
    return gmms

def predict_gmm(array, gmms):
    """ Return name of the matching GMM for each row in array """
    groups = gmms.keys()
    scores = numpy.vstack(map(lambda grp: gmms[grp].score(array), groups))
    return map(lambda idx: groups[idx], numpy.argmax(scores, axis=0))
