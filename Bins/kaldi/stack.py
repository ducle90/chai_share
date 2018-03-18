import chaipy.common as common
import chaipy.io as io
import numpy as np

from chaipy.data.temporal import TemporalData
from chaipy.kaldi import print_matrix


def stack_data(data, stack_frames, skip_frames):
    i = 0
    stacked = []
    stacked_dim = data.shape[1] * stack_frames
    while i < data.shape[0]:
        temp = list(data[i:i+stack_frames].flatten())
        # Pad if necessary
        if len(temp) < stacked_dim:
            frames_to_pad = (stacked_dim - len(temp)) / data.shape[1]
            temp += list(data[-1]) * frames_to_pad
        stacked.append(temp)
        i += skip_frames
    return np.array(stacked, dtype=data.dtype)


def main(args):
    # Check and set variables
    common.CHK_GT(args.frames_to_stack, 0)
    if args.frames_to_skip is None:
        args.frames_to_skip = args.frames_to_stack
    common.CHK_GT(args.frames_to_skip, 0)

    ds = TemporalData.from_kaldi(args.scp)
    io.log('Loaded dataset containing {} utts'.format(len(ds.get_utt_names())))

    io.log('Outputting stacked features (stack: {}, skip: {}) to stdout...'.format(
        args.frames_to_stack, args.frames_to_skip
    ))
    for utt_name in ds.get_utt_names():
        data = ds.get_data_by_utt_name(utt_name)
        stacked = stack_data(data, args.frames_to_stack, args.frames_to_skip)
        print_matrix(utt_name, stacked)


if __name__ == '__main__':
    desc = 'Takes in a Kaldi scp, stack features, and output stacked ' + \
           'features to stdout in Kaldi-compatible format.'
    parser = common.init_argparse(desc)
    parser.add_argument('scp', help='Kaldi scp')
    parser.add_argument('frames_to_stack', type=int,
                        help='How many frames to stack')
    parser.add_argument('--frames-to-skip', type=int, default=None,
                        help='How many frames to skip. If not set, skip ' + \
                             'the same number of frames stacked.')
    main(parser.parse_args())
