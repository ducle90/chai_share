import numpy
import chaipy.common as common
import chaipy.io as io

def main():
    desc = 'Reads in a pdf alignment and output prior counts to disk.'
    parser = common.init_argparse(desc)
    parser.add_argument('alipdf', help='pdf alignment file.')
    parser.add_argument('output_fname', help='File to output prior counts to')
    parser.add_argument('--num-pdfs', type=int, help='Number of pdfs. ' + \
                        'If not set, use max value in `alipdf`.')
    args = parser.parse_args()

    alipdf = io.dict_read(args.alipdf)
    pdfs = []
    for utt in alipdf.keys():
        pdfs.extend(numpy.asarray(alipdf[utt], dtype=numpy.int))
    bins = numpy.bincount(pdfs, minlength=args.num_pdfs)

    fw = open(args.output_fname, 'w')
    fw.write('[ {} ]\n'.format(' '.join(numpy.asarray(bins, dtype=numpy.str))))
    fw.close()

if __name__ == '__main__':
    main()
