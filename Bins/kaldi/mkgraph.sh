#!/bin/bash
set -e

##################################################
# Make decoding graph (must run before decoding) #
##################################################

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"

#----------------- COMMAND-LINE ARGUMENTS ------------------#
opts=           # Options to pass to mkgraph.sh

#----------------- MAIN PROGRAM -------------------#
. $UTILS_PATH/parse_options.sh
if [ $# != 3 ]; then
    echo "Usage: $0 [options] <lang-dir> <model-dir> <output-dir>"
    echo ""
    echo "Main options (for others, see top of script file)"
    echo "  --opts <opts>     # mkgraph options, e.g. '--mono'"
    exit 1
fi
lang=$1
model=$2
output=$3

where=`pwd`
cd $lang && lang_dir=`pwd` && cd $where
cd $model && model_dir=`pwd` && cd $where
mkdir -p $output
cd $output && output_dir=`pwd` && cd $where

cd $UTILS_PATH/..
./utils/mkgraph.sh $opts $lang_dir $model_dir $output_dir
