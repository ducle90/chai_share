import numpy as np
import chaipy.common as common
import chaipy.io as io
from chaipy.kaldi import ivector_ark_read, print_matrix
from chaipy.data.temporal import TemporalData, BufferedVarUttData

from nnet_forward import get_log_priors

# keras imports
from keras.models import model_from_json

def init_dataset(scp, context, padding, nutts, delay, ivectors):
    ds = TemporalData.from_kaldi(scp, context=context, padding=padding)
    if ivectors is not None:
        ds.set_utt_feats(utt_feats_dict=ivectors)
    buf_ds = BufferedVarUttData(
        ds, nutts=nutts, nframes=None, delay=delay, length_bins=1, shuffle=False
    )
    return buf_ds

def main():
    desc = 'Outputs Kaldi-compatible log-likelihood to stdout using a ' + \
           'Keras model. This mimics the design of Kaldi nnet-forward.'
    parser = common.init_argparse(desc)
    parser.add_argument('model_json', help='JSON description of the model')
    parser.add_argument('model_weights', help='File containing model weights')
    parser.add_argument('feats_scp', help='scp of input features')
    parser.add_argument('--context', type=int, default=8,
                        help='Number of context frames for splicing')
    parser.add_argument('--padding', default='replicate',
                        help='What to do with out-of-bound frames. Valid ' + \
                             'values: [replicate|zero]')
    parser.add_argument('--primary-task', type=int,
                        help='Set to enable multi-task model decoding')
    parser.add_argument('--nutts', type=int, default=10,
                        help='How many utterances to feed to the model at once')
    parser.add_argument('--delay', type=int, default=5,
                        help='Output delay in frames')
    parser.add_argument('--class-frame-counts', help='Kaldi vector with ' + \
                        'frame-counts of pdfs to compute log-priors')
    parser.add_argument('--prior-floor', type=float, default=1e-10,
                        help='Flooring constant for prior probability, ' + \
                             'i.e. pdfs with prior smaller than this ' + \
                             'value will be ignored during decoding.')
    parser.add_argument('--ivectors', help='Utterance i-vectors to append')
    args = parser.parse_args()

    io.log('Initializing dataset')
    ivectors = None if args.ivectors is None else \
            ivector_ark_read(args.ivectors, dtype=np.float32)
    buf_ds = init_dataset(
        args.feats_scp, args.context, args.padding,
        args.nutts, args.delay, ivectors
    )
    io.log('Initializing model')
    json_str = io.json_load(args.model_json)
    model = model_from_json(json_str)
    model.load_weights(args.model_weights)
    io.log('Initializing priors')
    log_priors = get_log_priors(args.class_frame_counts, args.prior_floor)
    if args.primary_task is not None:
        io.log('Multi-task decoding enabled, primary task {}'.format(args.primary_task))

    io.log('** Begin outputting **')
    while True:
        # Load data chunk
        chunk = buf_ds.read_next_chunk()
        if chunk is None:
            break
        Xs, _, eobs, utt_indices = chunk
        X = Xs[0]
        eob = eobs[0]
        utt_names = buf_ds.dataset().get_utt_names_by_utt_indices(utt_indices)
        y = model.predict(X, batch_size=len(utt_indices), verbose=0)
        if args.primary_task is not None:
            y = y[args.primary_task]
        y = np.log(y, y)
        if log_priors is not None:
            y -= log_priors
        for i in range(len(utt_indices)):
            print_matrix(utt_names[i], y[i][buf_ds.get_delay():eob[i]])

if __name__ == '__main__':
    main()
