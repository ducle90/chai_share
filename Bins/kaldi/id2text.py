import chaipy.common as common
import chaipy.io as io

def main():
    desc = 'Kaldi outputs token IDs in numbers. We can map them back to ' + \
            'textual form given an ID to text mapping. Will output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('fname', help='File to process. We expect each line ' + \
                        'to have tokens separated by whitespace, where ' + \
                        'the first token is a key or name (e.g. utt name) ' + \
                        'that can be skipped, and the rest are ID numbers.')
    parser.add_argument('id_map', help='Mapping from textual form to ID. ' + \
                        'We expect each line to have two tokens separated ' + \
                        'by whitespace, where the first token is the text ' + \
                        'and the second token is the ID number.')
    args = parser.parse_args()

    id_map = common.make_reverse_index(io.dict_read(args.id_map))
    # Check that mapping from number to text is 1-to-1
    for k in id_map.keys():
        if len(id_map[k]) != 1:
            raise ValueError('Mapping at {} not 1-1: {}'.format(k, id_map[k]))
        id_map[k] = id_map[k][0]

    with open(args.fname, 'r') as f:
        for line in f:
            ary = line.strip().split()
            for i in range(1, len(ary)):
                ary[i] = id_map[ary[i]]
            print ' '.join(ary)

if __name__ == '__main__':
    main()
