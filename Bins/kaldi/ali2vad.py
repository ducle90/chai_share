import chaipy.common as common
import chaipy.io as io
from chaipy.kaldi import ali_with_length_read, print_vector

def main():
    desc = 'Use phone alignment to generate VAD vectors. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('ali_phones_with_length', 
                        help='File containing phone alignment with length ' + \
                        '(generated with ali-to-phones --write-lengths=true)')
    parser.add_argument('silphones', help='List of phones regarded as silence')
    args = parser.parse_args()

    silphones = set(io.read_lines(args.silphones))
    io.log('{} silence phones: {}'.format(len(silphones), ':'.join(silphones)))
    alis = ali_with_length_read(
        args.ali_phones_with_length, ordered=True, expand=False
    )
    io.log('Loaded {} alignments'.format(len(alis)))

    for key in alis:
        vad = []
        for ali in alis[key]:
            phone, length = ali
            vad.extend([0.0 if phone in silphones else 1.0] * length)
        print_vector(key, vad)

if __name__ == '__main__':
    main()
