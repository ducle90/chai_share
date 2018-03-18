import theano
import numpy as np
import math

import chaipy.common as common
import chaipy.io as io
from chaipy.kaldi import ivector_ark_read, print_matrix
from chaipy.data import get_num_items
from chaipy.data.temporal import TemporalData
from chaipy.learning.pdnn import load_dnn

FLT_MAX = np.finfo(theano.config.floatX).max

def init_dataset(scp, context, padding, ivectors):
    dataset = TemporalData.from_kaldi(scp, context=context, padding=padding)
    if ivectors is not None:
        dataset.set_utt_feats(utt_feats_dict=ivectors)
    return dataset

def get_log_priors(class_frame_counts, prior_floor):
    if class_frame_counts is None:
        return None
    with open(class_frame_counts, 'r') as f:
        count_str = f.readline().strip().replace('[', '').replace(']', '')
    priors = np.asarray(count_str.split(), dtype=np.float32)
    priors = priors / np.sum(priors)
    # Add a small value before doing log to avoid NaN
    log_priors = np.log(priors + 1e-20)
    # Floor pdf
    num_floored = 0
    for i in range(len(log_priors)):
        if priors[i] < prior_floor:
            log_priors[i] = math.sqrt(FLT_MAX)
            num_floored += 1
    io.log('Floored {} pdf-priors (hard-set to {}, which disables DNN output)'.format(
        num_floored, math.sqrt(FLT_MAX)
    ))
    return log_priors

def __nnet_fwd(output_fn, dataset, x, shared_x,
               utt_names, utt_frames, log_priors):
    # Set data
    feats = dataset.get_data_by_utt_names(utt_names)
    common.CHK_EQ(len(feats), np.sum(utt_frames))
    x[:len(feats)] = feats
    shared_x.set_value(x[:len(feats)], borrow=True)
    # Process each utterance
    start = 0
    for utt, frames in zip(utt_names, utt_frames):
        end = start + frames
        log_probs = np.log(output_fn(start, end))
        if log_priors is not None:
            log_probs -= log_priors
        print_matrix(utt, log_probs)
        start = end

def main():
    desc = 'Outputs Kaldi-compatible log-likelihood to stdout using a pdnn ' + \
           'model. This mimics the design of Kaldi nnet-forward. Use this ' + \
           'for networks that cannot be converted to Kaldi, e.g. factored model'
    parser = common.init_argparse(desc)
    parser.add_argument('model_in', help='Model that can be read by load_dnn')
    parser.add_argument('feats_scp', help='scp of input features')
    parser.add_argument('--context', type=int, default=8,
                        help='Number of context frames for splicing')
    parser.add_argument('--padding', default='replicate',
                        help='What to do with out-of-bound frames. Valid ' + \
                             'values: [replicate|zero]')
    parser.add_argument('--class-frame-counts', help='Kaldi vector with ' + \
                        'frame-counts of pdfs to compute log-priors')
    parser.add_argument('--prior-floor', type=float, default=1e-10,
                        help='Flooring constant for prior probability, ' + \
                             'i.e. pdfs with prior smaller than this ' + \
                             'value will be ignored during decoding.')
    parser.add_argument('--ivectors', help='Utterance i-vectors to append')
    parser.add_argument('--chunk-size', default='300m',
                        help='Chunk size for data buffering')
    args = parser.parse_args()

    io.log('Initializing dataset')
    ivectors = None if args.ivectors is None else \
            ivector_ark_read(args.ivectors, dtype=theano.config.floatX)
    dataset = init_dataset(args.feats_scp, args.context, args.padding, ivectors)
    io.log('Initializing model')
    dnn = load_dnn(args.model_in)
    io.log('Initializing priors')
    log_priors = get_log_priors(args.class_frame_counts, args.prior_floor)

    # Initializing shared_ds according to chunk_size
    num_items = get_num_items(args.chunk_size, theano.config.floatX)
    max_frames = num_items / dataset.get_dim()
    max_utt_frames = np.max(map(dataset.get_num_frames_by_utt_name,
                                dataset.get_utt_names()))
    common.CHK_GE(max_frames, max_utt_frames)
    x = np.zeros((max_frames, dataset.get_dim()), dtype=theano.config.floatX)
    shared_x = theano.shared(x, name='x', borrow=True)
    io.log('Using shared_x with size {} ({})'.format(x.shape, args.chunk_size))
    io.log('...getting output function')
    output_fn = dnn.build_output_function(shared_x)
    io.log('Got it!')

    io.log('** Begin outputting **')
    utt_names, utt_frames, total_frames = [], [], 0
    for utt in dataset.get_utt_names():
        frames = dataset.get_num_frames_by_utt_name(utt)
        if total_frames + frames > max_frames:
            __nnet_fwd(output_fn, dataset, x, shared_x,
                       utt_names, utt_frames, log_priors)
            utt_names, utt_frames, total_frames = [], [], 0
        utt_names.append(utt)
        utt_frames.append(frames)
        total_frames += frames
    __nnet_fwd(output_fn, dataset, x, shared_x,
               utt_names, utt_frames, log_priors)

if __name__ == '__main__':
    main()
