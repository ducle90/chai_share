import chaipy.common as common
import chaipy.io as io

from chaipy.data.temporal import TemporalData


def main(args):
    ds = TemporalData.from_kaldi(args.scp)
    io.log('Loaded dataset containing {} utts'.format(len(ds.get_utt_names())))
    utt2label = io.dict_read(args.utt2label)
    io.log('Loaded utt2label containing {} entries'.format(len(utt2label)))

    for utt_name in ds.get_utt_names():
        if utt_name not in utt2label:
            io.log('WARNING: {} not in utt2label, skipping'.format(utt_name))
        lbl = utt2label[utt_name]
        dur = ds.get_num_frames_by_utt_name(utt_name)
        print '{} {}'.format(utt_name, ' '.join([lbl] * dur))


if __name__ == '__main__':
    desc = 'Takes in a Kaldi scp and utterance-level labels, outputs ' + \
           'frame-level labels of all utterances in the scp to stdout. ' + \
           'Utterances that are in the scp but not in the label mapping ' + \
           'will be skipped.'
    parser = common.init_argparse(desc)
    parser.add_argument('scp', help='Kaldi scp')
    parser.add_argument('utt2label', help='Mapping from utterance to label')
    main(parser.parse_args())

