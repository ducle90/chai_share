import chaipy.common as common
import chaipy.io as io
from chaipy.kaldi import ivector_ark_read, print_vector

def main():
    desc = 'Convert from speaker i-vectors to utt-ivectors. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('spk_ivectors', help='File containing spk i-vectors.')
    parser.add_argument('utt2spk', help='Kaldi utt2spk mapping.')
    args = parser.parse_args()

    spk_ivectors = ivector_ark_read(args.spk_ivectors)
    utt2spk = io.dict_read(args.utt2spk, ordered=True)
    spk2utt = common.make_reverse_index(utt2spk, ordered=True)

    wrote = 0
    for spk in spk2utt.keys():
        for utt in spk2utt[spk]:
            print_vector(utt, spk_ivectors[spk])
            wrote += 1
    io.log('Wrote {} utt i-vectors for {} spks'.format(wrote, len(spk2utt)))

if __name__ == '__main__':
    main()
