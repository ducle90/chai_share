import chaipy.common as common
import chaipy.io as io
from chaipy.kaldi import ali_with_length_read

def main():
    desc = 'Convert from alignment with length to regular alignments. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('ali_with_length', help='Alignment with lengths')
    args = parser.parse_args()

    ali = ali_with_length_read(args.ali_with_length, ordered=True, expand=True)
    io.log('Read {} aligment with lengths'.format(len(ali)))

    for key in ali:
        print '{} {}'.format(key, ' '.join(ali[key]))

if __name__ == '__main__':
    main()
