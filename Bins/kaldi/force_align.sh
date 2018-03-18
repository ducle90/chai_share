#!/bin/bash

##############################################
# Perform forced alignment using GMM model.  #
# Script adapted from Kaldi's WSJ recipe.    #
##############################################

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"

#----------------- COMMAND-LINE ARGUMENTS ------------------#
nj=4            # Number of parallel jobs
cmd="$UTILS_PATH/run.pl"
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=20
retry_beam=80
careful=false
compress=true

#----------------- MAIN PROGRAM -------------------#
. $UTILS_PATH/parse_options.sh
if [ $# != 4 ]; then
    echo "Usage: $0 [options] <data-dir> <lang-dir> <model-dir> <output-dir>"
    echo ""
    echo "Main options (for others, see top of script file)"
    echo "  --nj <nj>               # Number of parallel jobs"
    echo "  --beam <beam>           # Beam width to use"
    echo "  --retry-beam <beam>     # Beam width for retrying"
    echo "  --compress <true|false> # Whether or not to compress alignments"
    exit 1
fi
data=$1
lang=$2
model=$3
output_dir=$4
logs_dir=$output_dir/logs

for f in $data/text $lang/oov.int $model/tree $model/final.mdl; do
    [ ! -f $f ] && echo "Expected file $f to exist" && exit 1
done

oov=`cat $lang/oov.int` || exit 1
mkdir -p $logs_dir
echo $nj >$output_dir/num_jobs
sdata=$data/split${nj}
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || \
    (where=`pwd` && cd $data && data_path=`pwd` && cd $UTILS_PATH/.. && \
        utils/split_data.sh $data_path $nj && cd $where) || exit 1

feats="scp:$sdata/JOB/feats.scp"

echo "Aligning data in $data using GMM model from $model, outputting to $output_dir"

tra="ark:$UTILS_PATH/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|"
if [ "$compress" == "true" ]; then
    ali_output="ark,t:|gzip -c >$output_dir/ali.JOB.gz"
else
    ali_output="ark,t:$output_dir/ali.JOB.txt"
fi
$cmd JOB=1:$nj $logs_dir/align.JOB.log compile-train-graphs \
    $model/tree $model/final.mdl $lang/L.fst "$tra" ark:- \| gmm-align-compiled \
    $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful \
    $model/final.mdl ark:- "$feats" "$ali_output" || exit 1;

echo "Done aligning data"
