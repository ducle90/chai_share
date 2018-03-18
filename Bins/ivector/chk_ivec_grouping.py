# This script is intended to perform a sanity check on i-vectors. We provide
# a manual grouping of these i-vectors, e.g. group by speaker, gender, age,
# etc... For each group we will fit a 1-component GMM. Then, we will do
# classification from i-vector to group based on the GMM that yields the
# highest probability score for this i-vector. Summary of classification
# performance will be printed to stdout. Progress is printed to stderr. This 
# script takes a list of keys to form the training set. Everything else is test.

import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score
import chaipy.io as io
import chaipy.common as common
from chaipy.kaldi import ivector_ark_read
from chaipy.learning.gmm import train_gmms, predict_gmm
from chaipy.data import partition
from chaipy.common.metrics import comp_UARs

def run(ivectors, train_group2ivecs, test_ivecs, cov_type):
    all_data = OrderedDict()
    for group in train_group2ivecs:                                               
        train_data = np.asarray(map(lambda x: ivectors[x], 
                                    train_group2ivecs[group]))
        all_data[group] = train_data
    io.log('Training GMMs')
    gmms = train_gmms(all_data, n_components=1, cov_type=cov_type, ordered=True)
    io.log('Getting predictions')
    test_data = np.vstack(map(lambda x: ivectors[x], test_ivecs))
    return predict_gmm(test_data, gmms)

def main():
    import argparse
    desc = 'Perform sanity check for i-vector grouping'
    parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('ivector_ark', help='i-vectors ark file')
    parser.add_argument('ivec2group', help='i-vector to group mapping')
    parser.add_argument('train_keys', help='Keys for training set')
    parser.add_argument('--cov-type', default='diag',
                        help='GMM covariance type (full|tied|diag|spherical)')
    parser.add_argument('--ignore', nargs='+', default=[],
                        help='Ignore these groups')
    parser.add_argument('--test-keys', help='Keys for test set. If not ' + \
                        'specified, treat keys not in training set as test.')
    args = parser.parse_args()

    io.log('Reading i-vector ark from {}'.format(args.ivector_ark))
    ivectors = ivector_ark_read(args.ivector_ark)
    io.log('Reading i-vector grouping from {}'.format(args.ivec2group))
    ivec2group = io.dict_read(args.ivec2group)
    io.log('Ignore list: {}'.format(args.ignore))
    all_keys = [x for x in ivec2group if ivec2group[x] not in args.ignore]
    io.log('Reading training keys from {}'.format(args.train_keys))
    train_keys = set(io.read_lines(args.train_keys))
    test_keys = None
    if args.test_keys is not None:
        io.log('Reading test keys from {}'.format(args.test_keys))
        test_keys = set(io.read_lines(args.test_keys))
    
    train_ivec2group, test_ivec2group = OrderedDict(), OrderedDict()
    for k in all_keys:
        if k in ivectors:
            if k in train_keys:
                train_ivec2group[k] = ivec2group[k]
            elif test_keys is None or k in test_keys:
                test_ivec2group[k] = ivec2group[k]
    test_keys = test_ivec2group.keys()
    io.log('Train: {}, Test: {}'.format(len(train_keys), len(test_keys)))
    train_group2ivecs = common.make_reverse_index(train_ivec2group)

    io.log('GMM covariance type: {}'.format(args.cov_type))
    # Fit GMM and do prediction
    corr_lbls = map(lambda x : test_ivec2group[x], test_keys)
    pred_lbls = run(ivectors, train_group2ivecs, test_keys, args.cov_type)
    # Report results
    acc = 100 * accuracy_score(corr_lbls, pred_lbls)
    print 'Overall accuracy: {:.2f} (%)'.format(acc)
    group_accs, group_names = comp_UARs(corr_lbls, pred_lbls)
    print 'Mean per-group accuracy: {:.2f} (%)'.format(100 * np.mean(group_accs))
    print 'Individual group accuracies:'
    for group_acc, group_name in zip(group_accs, group_names):
        print '\t{} - {} (%)'.format(group_name, 100 * group_acc)

if __name__ == '__main__':
    main()
