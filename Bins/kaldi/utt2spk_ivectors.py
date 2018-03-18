import chaipy.common as common
import chaipy.io as io
from chaipy.kaldi import ivector_ark_read, print_vector

def main():
    desc = 'Convert from utt i-vectors to spk-ivectors. NOTE: this ' + \
           'script does not check the values of utt i-vectors that belong ' + \
           'to the same spk. It will simply treat the first utt i-vector ' + \
           'it finds from a spk as the i-vector for that spk. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('utt_ivectors', help='File containing utt i-vectors.')
    parser.add_argument('utt2spk', help='Kaldi utt2spk mapping.')
    args = parser.parse_args()

    utt_ivectors = ivector_ark_read(args.utt_ivectors, ordered=True)
    utt2spk = io.dict_read(args.utt2spk)

    processed_spks = set()
    for utt in utt_ivectors.keys():
        spk = utt2spk[utt]
        if spk in processed_spks:
            continue
        print_vector(spk, utt_ivectors[utt])
        processed_spks.add(spk)
    io.log('Wrote {} spk i-vectors'.format(len(processed_spks)))

if __name__ == '__main__':
    main()
