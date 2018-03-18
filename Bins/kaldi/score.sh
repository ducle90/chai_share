#!/bin/bash
set -e

############################
# Score recognition output #
############################

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"

#----------------- COMMAND-LINE ARGUMENTS ------------------#
opts=       # Options to pass to score.sh

#----------------- MAIN PROGRAM -------------------#
. $UTILS_PATH/parse_options.sh
if [ $# != 3 ]; then
    echo "Usage: $0 [options] <data-dir> <lang-dir|graph-dir> <decode-dir>"
    echo ""
    echo "Main options (for others, see top of script file)"
    echo "  --opts <opts>   # score options, e.g. '--min-lmwt 9 --max-lmwt 20'"
    exit 1
fi
data=$1
lang_or_graph=$2
decode=$3

where=`pwd`
cd $data && data_dir=`pwd` && cd $where
cd $lang_or_graph && lang_or_graph_dir=`pwd` && cd $where
cd $decode && decode_dir=`pwd` && cd $where

cd $UTILS_PATH/..
./local/score.sh $opts $data_dir $lang_or_graph_dir $decode_dir
