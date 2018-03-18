import chaipy.common as common
import chaipy.io as io
from chaipy.kaldi import ali_with_length_read

def get_phone_tokens(alis, id2phone, sil_phones):
    """ Convert each entry in alis into a phone token. Phone IDs are converted
    to phone names, and silence phones will not be included.
    """
    phone_tokens = []
    offset = 0
    for ali in alis:
        phone_id, length = ali
        if phone_id not in sil_phones:
            phone_tokens.append(
                common.TimeTag(id2phone[phone_id], offset, offset + length)
            )
        offset += length
    return (phone_tokens, offset)

def phone2word_ali(key, phone_tokens, words, lexicon, sil_label, length):
    """ Print word alignment to stdout. If there are several possible
    alignments, arbitrarily choose the first one.
    """
    word_alis = get_alignments(words, phone_tokens, lexicon)
    if word_alis is None:
        raise ValueError('{} - failed to align {} to {}'.format(
            key, words, phone_tokens
        ))
    if len(word_alis) > 1:
        io.log('WARNING - {} has multiple ({}) alignments: {} ({})'.format(
            key, len(word_alis), word_alis, words
        ))
    __print_ali(key, word_alis[0], sil_label, length)

def get_alignments(words, phone_tokens, lexicon):
    """ Return a list of possible alignments when aligning `words` to
    `phone_tokens`. Return `None` if no alignment can be found.
    """
    # Base case 1: success
    if len(words) == 0 and len(phone_tokens) == 0:
        return [[]]
    # Base case 2: failure
    elif len(words) == 0 or len(phone_tokens) == 0:
        return None
    # Recursive case
    else:
        alignments = []
        indices = __find_matches(lexicon[words[0]], phone_tokens)
        if len(indices) == 0:
            return None
        for idx in indices:
            word_token = (words[0], phone_tokens[:idx])
            sub_alignments = get_alignments(words[1:], phone_tokens[idx:], lexicon)
            if sub_alignments is not None:
                for sub_ali in sub_alignments:
                    ali = [word_token]
                    ali.extend(sub_ali)
                    alignments.append(ali)
        return alignments if len(alignments) > 0 else None

def __find_matches(pronunciations, phone_tokens):
    """ Return indices of tokens that match one of the pronunciations. For
    example, if `pronunciations = [['a', 'b'], ['a', 'c'], ['a', 'b', 'c']]`
    and `phone_tokens = ['a', 'b', 'c', 'd', 'e']`, return `[2, 3]`.
    If no match can be found, return an empty list.
    """
    matches = []
    for pron in pronunciations:
        if len(pron) <= len(phone_tokens):
            cand = [token.name for token in phone_tokens[:len(pron)]]
            if all([pron[i] == cand[i] for i in range(len(pron))]):
                matches.append(len(pron))
    # Sanity check
    if len(matches) != len(set(matches)):
        raise ValueError('Duplicate in matches: {}, {} ({})'.format(
            matches, pronunciations, phone_tokens
        ))
    return matches

def __print_sil(sil_label, start, end):
    print '{} {} {} 0.0 {}'.format(start, end, sil_label, sil_label)

def __print_ali(key, word_tokens, sil_label, length):
    """ Output format roughly follows that of HTK.
    """
    print '"{}"'.format(key)
    cursor = 0
    for word_token in word_tokens:
        word, phone_tokens = word_token
        if cursor < phone_tokens[0].start:
            __print_sil(sil_label, cursor, phone_tokens[0].start)
        is_first = True
        for tok in phone_tokens:
            wrd_label = ' {}'.format(word) if is_first else ''
            print '{} {} {} 0.0{}'.format(tok.start, tok.end, tok.name, wrd_label)
            is_first = False
        cursor = phone_tokens[-1].end
    if cursor < length:
        __print_sil(sil_label, cursor, length)
    print '.'

def main():
    desc = 'Convert phone to word alignment. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('ali_phones_with_length',
                        help='File containing phone alignment with length ' + \
                        '(generated with ali-to-phones --write-lengths=true)')
    parser.add_argument('text', help='Kaldi word-level transcript')
    parser.add_argument('phone_map', help='Mapping from text to phone ID. ' + \
                        'We expect each line to have two tokens separated ' + \
                        'by whitespace, where the first token is the phone ' + \
                        'and the second token is the ID number.')
    parser.add_argument('lexicon', help='Pronunciation lexicon')
    parser.add_argument('--sil-phones', nargs='+', default=[],
                        help='IDs of phones regarded as silence')
    parser.add_argument('--sil-label', default='sil',
                        help='Label of silence phone/word to use in output')
    args = parser.parse_args()

    alis = ali_with_length_read(
        args.ali_phones_with_length, ordered=True, expand=False
    )
    io.log('Loaded {} alignments'.format(len(alis)))
    text = io.dict_read(args.text, lst=True)
    io.log('Loaded transcript containing {} utterances'.format(len(text)))
    phone2id = io.dict_read(args.phone_map)
    io.log('Loaded phone2id containing {} phones'.format(len(phone2id)))
    id2phone = {}
    # We normalize the phone name so that IDs of phone variants will map to
    # the primary phone. For example, IDs of sil, sil_B, sil_E, sil_I, sil_S
    # will all map to sil. The assumption here is that anything after and
    # including the '_' character is not part of the primary phone name.
    for phone in phone2id.keys():
        nphone = phone.split('_')[0]
        id2phone[phone2id[phone]] = nphone
    io.log('Total phones in id2phone: {}'.format(len(set(id2phone.values()))))
    lexicon = io.lexicon_read(args.lexicon)
    io.log('Loaded lexicon containing {} words'.format(len(lexicon)))
    sil_phones = set(args.sil_phones)
    io.log('sil_phones: {} ({}), sil_label: {}'.format(
        sil_phones, [id2phone[i] for i in sil_phones], args.sil_label
    ))

    for key in alis:
        phone_tokens, length = get_phone_tokens(alis[key], id2phone, sil_phones)
        if len(phone_tokens) == 0:
            io.log('WARNING: {} - no non-silence tokens'.format(key))
            continue
        if key not in text:
            io.log('WARNING: {} not in text'.format(key))
            continue
        phone2word_ali(
            key, phone_tokens, text[key], lexicon, args.sil_label, length
        )

if __name__ == '__main__':
    main()
