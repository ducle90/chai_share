# This script is intended to perform a sanity check on i-vectors. We provide
# a manual grouping of these i-vectors, e.g. group by speaker, gender, age,
# etc... For each group we will fit a 1-component GMM. Then, we will do
# classification from i-vector to group based on the GMM that yields the
# highest probability score for this i-vector. Summary of classification
# performance will be printed to stdout. Progress is printed to stderr.
# This script performs leave-one-speaker-out cross-validation. For N-fold
# CV, look for chk_ivec_grouping_nfold.py.

import traceback
import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score, confusion_matrix

import chaipy.io as io
import chaipy.common as common
from chaipy.kaldi import ivector_ark_read
from chaipy.learning.gmm import train_gmms, predict_gmm
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

def report(corr_lbls, pred_lbls, spks=None):
    # Report overall results
    acc = 100 * accuracy_score(corr_lbls, pred_lbls)
    print 'Overall accuracy: {} (%)'.format(acc)
    group_accs, group_names = comp_UARs(corr_lbls, pred_lbls)
    print 'Mean per-group accuracy: {} +/- {} (%)'.format(
            100 * np.mean(group_accs), 100 * np.std(group_accs)
    )
    print 'Individual group accuracies:'
    for group_acc, group_name in zip(group_accs, group_names):
        print '\t{} - {} (%)'.format(group_name, 100 * group_acc)
    # Report speaker results
    if spks is None:
        return
    spk2lbls = {}
    for corr_lbl, pred_lbl, spk in zip(corr_lbls, pred_lbls, spks):
        if spk not in spk2lbls:
            spk2lbls[spk] = []
        spk2lbls[spk].append([corr_lbl, pred_lbl])
    spk_accs = []
    for spk in np.sort(spk2lbls.keys()):
        lbl_data = np.asarray(spk2lbls[spk])
        spk_accs.append(accuracy_score(lbl_data[:,0], lbl_data[:,1]))
    print 'Mean per-speaker accuracy: {} +/- {} (%)'.format(
            100 * np.mean(spk_accs), 100 * np.std(spk_accs)
    )
    print 'Individual speaker accuracies:'
    for spk, acc in zip(np.sort(spk2lbls.keys()), spk_accs):
        print '\t{} - {} (%)'.format(spk, 100 * acc)
    # Confusion matrix
    print 'Confusion matrix: (rows = correct, columns = prediction)'
    labels = np.sort(np.unique(corr_lbls))
    cmat = confusion_matrix(corr_lbls, pred_lbls, labels=labels)
    cmat = np.asarray(cmat, dtype=np.str)
    print '\tCMat\t{}'.format('\t'.join(labels))
    for i in range(len(labels)):
        print '\t{}\t{}'.format(labels[i], '\t'.join(cmat[i]))

def main():
    import argparse
    desc = 'Perform sanity check for i-vector grouping'
    parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('ivector_ark', help='i-vectors ark file')
    parser.add_argument('ivec2group', help='i-vector to group mapping')
    parser.add_argument('ivec2spk', help='i-vector to speaker mapping')
    parser.add_argument('--cov-type', default='diag',
                        help='GMM covariance type (full|tied|diag|spherical)')
    args = parser.parse_args()

    io.log('Reading i-vector ark from {}'.format(args.ivector_ark))
    ivectors = ivector_ark_read(args.ivector_ark)
    io.log('Reading i-vector grouping from {}'.format(args.ivec2group))
    ivec2group = io.dict_read(args.ivec2group)
    group2ivecs = common.make_reverse_index(ivec2group)
    io.log('Reading i-vector to speaker mapping from {}'.format(args.ivec2spk))
    ivec2spk = io.dict_read(args.ivec2spk)
    spk2ivecs = common.make_reverse_index(ivec2spk)

    io.log('GMM covariance type: {}'.format(args.cov_type))
    spks = []
    corr_lbls = []
    pred_lbls = []
    for i, spk in enumerate(spk2ivecs):
        io.log('--- Held-out spk: {} ({} / {}) ---'.format(spk, i+1, len(spk2ivecs)))
        # Common base for training i-vectors
        train_group2ivecs = OrderedDict()
        for other_spk in spk2ivecs:
            if other_spk == spk:
                continue
            for ivec in spk2ivecs[other_spk]:
                group = ivec2group[ivec]
                if group not in train_group2ivecs:
                    train_group2ivecs[group] = []
                train_group2ivecs[group].append(ivec)
        # Get test i-vectors
        test_ivecs = spk2ivecs[spk]
        # Get results
        try:
            preds = run(ivectors, train_group2ivecs, test_ivecs, args.cov_type)
            spks.extend([spk] * len(test_ivecs))
            corr_lbls.extend(map(lambda x: ivec2group[x], test_ivecs))
            pred_lbls.extend(preds)
        except RuntimeError:
            traceback.print_exc()
            io.log('...skipping {}'.format(spk))

    # Report results
    report(corr_lbls, pred_lbls, spks)

if __name__ == '__main__':
    main()
