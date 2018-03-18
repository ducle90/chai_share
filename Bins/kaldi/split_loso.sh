#!/bin/bash
set -e

###########################################
# Split data using leave-one-speaker-out. #
###########################################

function split {
    if [ $# != 3 ]; then
        echo "Usage: split <spk> <data-dir> <fname>"
        exit 1
    fi
    spk=$1
    data=$2
    fname=$3

    grep "^$spk" $data/$fname >$data/LOSO/$spk/test/$fname
    grep -v "^$spk" $data/$fname >$data/LOSO/$spk/train/$fname
}

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"

#----------------- COMMAND-LINE ARGUMENTS ------------------#
extras=         # Extra files to split

#----------------- MAIN PROGRAM -------------------#
. $UTILS_PATH/parse_options.sh
if [ $# != 1 ]; then
    echo "Usage: $0 [options] <data-dir>"
    echo ""
    echo "Main options (for others, see top of script file)"
    echo "  --extras <str>      # Extra files to split, e.g. 'segments utts'"
    exit 1
fi
data=$1

for f in feats.scp spk2utt utt2spk $extras; do
    [ ! -f $data/$f ] && echo "No such file: $data/$f" && exit 1
done

echo "Splitting data and outputting to $data/LOSO"
mkdir -p $data/LOSO

for spk in `cut -f1 $data/spk2utt -d ' '`; do
    echo "...$spk"
    for f in train test; do mkdir -p $data/LOSO/$spk/$f; done
    # Standard files
    split $spk $data feats.scp
    split $spk $data spk2utt
    split $spk $data utt2spk
    if [ -f $data/text ]; then split $spk $data text; fi
    if [ -f $data/wav.scp ]; then split $spk $data wav.scp; fi
    if [ -f $data/cmvn.scp ]; then split $spk $data cmvn.scp; fi
    # Extra files
    for f in $extras; do split $spk $data $f; done
done
