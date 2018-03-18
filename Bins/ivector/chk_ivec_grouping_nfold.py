# This script is intended to perform a sanity check on i-vectors. We provide
# a manual grouping of these i-vectors, e.g. group by speaker, gender, age,
# etc... For each group we will fit a 1-component GMM. Then, we will do
# classification from i-vector to group based on the GMM that yields the
# highest probability score for this i-vector. Summary of classification
# performance will be printed to stdout. Progress is printed to stderr. This 
# script performs N-fold CV by withholding a fraction of data from each group.
# For leave-one-speaker-out CV, see chk_ivec_grouping_loso.py.

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
    parser.add_argument('--cov-type', default='diag',
                        help='GMM covariance type (full|tied|diag|spherical)')
    parser.add_argument('--withheld-frac', type=float,
                        help='Fraction of i-vectors retained from each group ' \
                                + 'for testing. If not set, use the same ' \
                                + 'data for training and testing.')
    args = parser.parse_args()

    io.log('Reading i-vector ark from {}'.format(args.ivector_ark))
    ivectors = ivector_ark_read(args.ivector_ark)
    io.log('Reading i-vector grouping from {}'.format(args.ivec2group))
    ivec2group = io.dict_read(args.ivec2group)
    group2ivecs = common.make_reverse_index(ivec2group)

    io.log('GMM covariance type: {}'.format(args.cov_type))
    io.log('Withheld fraction: {}'.format(args.withheld_frac))
    # Fit GMM and do prediction
    if args.withheld_frac is None:
        corr_lbls = map(lambda x: ivec2group[x], ivectors.keys())
        pred_lbls = run(ivectors, group2ivecs, ivectors.keys(), args.cov_type)
    else:
        corr_lbls = []
        pred_lbls = []
        for group in group2ivecs:
            # Common base for training i-vectors
            train_group2ivecs = OrderedDict()
            for other_group in group2ivecs:
                if other_group != group:
                    train_group2ivecs[other_group] = group2ivecs[other_group]
            # Get partitions of test i-vectors and step through each one
            test_partitions = partition(group2ivecs[group], args.withheld_frac)
            for i in range(len(test_partitions)):
                io.log('-- Partition {} / {} for {}'.format(
                    i + 1, len(test_partitions), group
                ))
                test_ivecs = test_partitions[i]
                # Get training i-vectors for this group
                train_ivecs = []
                for j in range(len(test_partitions)):
                    if j != i:
                        train_ivecs.extend(test_partitions[j])
                train_group2ivecs[group] = train_ivecs
                # Get results
                corr_lbls.extend(map(lambda x: ivec2group[x], test_ivecs))
                pred_lbls.extend(run(ivectors, train_group2ivecs,
                                     test_ivecs, args.cov_type))

    # Report results
    acc = 100 * accuracy_score(corr_lbls, pred_lbls)
    print 'Overall accuracy: {} (%)'.format(acc)
    group_accs, group_names = comp_UARs(corr_lbls, pred_lbls)
    print 'Mean per-group accuracy: {} (%)'.format(100 * np.mean(group_accs))
    print 'Individual group accuracies:'
    for group_acc, group_name in zip(group_accs, group_names):
        print '\t{} - {} (%)'.format(group_name, 100 * group_acc)

if __name__ == '__main__':
    main()
