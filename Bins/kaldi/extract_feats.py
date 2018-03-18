import theano
import numpy as np
import math

import chaipy.common as common
import chaipy.io as io
from chaipy.data import get_num_items
from chaipy.data.temporal import TemporalData
from chaipy.learning.pdnn import load_dnn

# pdnn import
from io_func.kaldi_feat import KaldiWriteOut

def init_dataset(scp, context, padding, ivectors):
    dataset = TemporalData.from_kaldi(scp, context=context, padding=padding)
    if ivectors is not None:
        dataset.set_utt_feats(utt_feats_dict=ivectors)
    return dataset

def main():
    desc = 'Extract features with DNN. Output to Kaldi ark.'
    parser = common.init_argparse(desc)
    parser.add_argument('model_in', help='Model that can be read by load_dnn')
    parser.add_argument('feats_scp', help='scp of input features')
    parser.add_argument('ark_out', help='Output ark file')
    parser.add_argument('--output-layer', type=int, default=-2,
                        help='Layer to use for extracting features. ' + \
                             'Negative index can be used. For example, ' + \
                             '-1 means the last layer, and so on.')
    parser.add_argument('--context', type=int, default=8,
                        help='Number of context frames for splicing')
    parser.add_argument('--padding', default='replicate',
                        help='What to do with out-of-bound frames. Valid ' + \
                             'values: [replicate|zero]')
    parser.add_argument('--ivectors', help='Utterance i-vectors to append')
    parser.add_argument('--chunk-size', default='300m',
                        help='Chunk size for data buffering')
    args = parser.parse_args()

    io.log('Initializing dataset')
    ivectors = None if args.ivectors is None else \
            io.ivector_ark_read(args.ivectors, dtype=theano.config.floatX)
    dataset = init_dataset(args.feats_scp, args.context, args.padding, ivectors)
    io.log('Initializing model')
    dnn = load_dnn(args.model_in)

    # Initializing shared_ds according to chunk_size
    num_items = get_num_items(args.chunk_size, theano.config.floatX)
    max_frames = num_items / dataset.get_dim()
    max_utt_frames = np.max(map(dataset.get_num_frames_by_utt_name,
                                dataset.get_utt_names()))
    common.CHK_GE(max_frames, max_utt_frames)
    x = np.zeros((max_frames, dataset.get_dim()), dtype=theano.config.floatX)
    io.log('...getting extraction function')
    extract_fn = dnn.build_extract_feat_function(args.output_layer)
    io.log('Got it!')

    io.log('** Begin outputting to {} **'.format(args.ark_out))
    ark_out = KaldiWriteOut(args.ark_out)
    utt_names, utt_frames, total_frames = [], [], 0
    for utt in dataset.get_utt_names():
        frames = dataset.get_num_frames_by_utt_name(utt)
        if total_frames + frames > max_frames:
            __extract(extract_fn, ark_out, dataset, x, utt_names, utt_frames)
            utt_names, utt_frames, total_frames = [], [], 0
        utt_names.append(utt)
        utt_frames.append(frames)
        total_frames += frames
    __extract(extract_fn, ark_out, dataset, x, utt_names, utt_frames)
    ark_out.close()

def __extract(extract_fn, ark_out, dataset, x, utt_names, utt_frames):
    # Set data
    feats = dataset.get_data_by_utt_names(utt_names)
    common.CHK_EQ(len(feats), np.sum(utt_frames))
    x[:len(feats)] = feats
    # Get big feature matrix
    ext_feats = extract_fn(x[:len(feats)])
    # Write to ark for each utterance
    start = 0
    for utt, frames in zip(utt_names, utt_frames):
        end = start + frames
        ark_out.write_kaldi_mat(utt, ext_feats[start:end])
        start = end

if __name__ == '__main__':
    main()
