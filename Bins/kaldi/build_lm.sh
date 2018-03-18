#!/bin/bash
set -e

################################################################
# Build n-gram LM and output in FST format.                    #
#                                                              #
# NOTE: this script assumes SRILM has already been installed.  #
# Try running `ngram-count -help`. If you see the help text,   #
# it means SRILM has been installed on this machine. If not,   #
# you can install it by running install_srilm.sh in            #
# /home/public/kaldi/tools. Then you need to add SRILM to      #
# your PATH variable. Here's an example:                       #
#                                                              #
#     export SRILM=$KD_ROOT/../tools/srilm                     #
#     export PATH=$PATH:$SRILM/bin:$SRILM/bin/i686-m64         #
################################################################

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"

#----------------- COMMAND-LINE PARAMETERS ----------------#
cmd="$UTILS_PATH/run.pl"
order=2                         # n-gram order

#----------------- MAIN PROGRAM -------------------#
. $UTILS_PATH/parse_options.sh
if [ $# != 2 ]; then
    echo "Usage: $0 <text> <words> <output_dir>"
    echo "    <text> - Each line is one utterance (without utt name)"
    echo "    <lang> - lang folder, G.fst will be written to here"
    echo ""
    echo "Configurable optional parameters:"
    echo "    --order [2] - Degree of n-gram (2 means bigram with backoff)"
    exit 1
fi
text=$1
lang=$2
tmp_dir=/tmp/build_lm.`date +%s`

for f in $text $lang/words.txt; do
    [ ! -f $f ] && echo "No such file: $f" && exit 1
done
mkdir -p $tmp_dir

cat $text | sed -e 's/\s\+/\n/g' | sort | uniq >$tmp_dir/vocab

echo "Generate n-gram counts and LM"
$cmd $lang/ngram-count.log ngram-count \
    -vocab $tmp_dir/vocab -text $text \
    -order $order -write $lang/G.count -lm $lang/G.arpa -unk -sort

echo "Find OOVs"
$UTILS_PATH/find_arpa_oovs.pl $lang/words.txt $lang/G.arpa >$tmp_dir/oovs_arpa.txt

echo "Convert ARPA LM to FST"
cat $lang/G.arpa | \
    grep -v '<s> <s>' | grep -v '</s> <s>' | grep -v '</s> </s>' | \
    arpa2fst - | fstprint | \
    $UTILS_PATH/remove_oovs.pl $tmp_dir/oovs_arpa.txt | \
    $UTILS_PATH/eps2disambig.pl | $UTILS_PATH/s2eps.pl | fstcompile \
    --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
    --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel >$lang/G.fst

rm -r $tmp_dir

echo "Validate lang folder" && cd $lang && lang_path=`pwd`
cd $UTILS_PATH/..
./utils/validate_lang.pl $lang_path
