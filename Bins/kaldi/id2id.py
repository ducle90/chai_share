import chaipy.common as common
import chaipy.io as io

def main():
    desc = 'Convert from one mapping to another. Will output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('fname', help='File to process. We expect each line ' + \
                        'to have tokens separated by whitespace, where ' + \
                        'the first token is a key or name (e.g. utt name) ' + \
                        'that can be skipped, and the rest are values.')
    parser.add_argument('id_map', help='Mapping from one ID to another ID. ' + \
                        'Each line has two tokens separated by whitespace.')
    args = parser.parse_args()

    id_map = io.dict_read(args.id_map)
    io.log('Read {} mappings'.format(len(id_map)))

    with open(args.fname, 'r') as f:
        for line in f:
            ary = line.strip().split()
            for i in range(1, len(ary)):
                ary[i] = id_map[ary[i]]
            print ' '.join(ary)

if __name__ == '__main__':
    main()
