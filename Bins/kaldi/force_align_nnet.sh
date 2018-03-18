#!/bin/bash

##############################################
# Perform forced alignment using nnet model. #
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
use_gpu="no"
device=
nnet_name="final.nnet"
ivector=

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
    echo "  --use-gpu <yes|no|optional>"
    echo "  --device <list>         # List of GPUs to consider, e.g. '0,1'."
    echo "  --nnet-name <str>       # Custom nnet name (default: final.nnet)"
    echo "  --ivector <str>         # Path to textual ark containing i-vectors"
    echo "                          # to append to features."
    exit 1
fi
data=$1
lang=$2
model=$3
output_dir=$4
logs_dir=$output_dir/logs

nnet=$model/$nnet_name
class_frame_counts=$model/prior_counts

for f in $data/text $lang/oov.int $model/tree $model/final.mdl $nnet $class_frame_counts; do
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

echo "Detect context size from data"
mdim=`nnet-info $nnet | grep "^input-dim" | sed -e 's/.*\s\+//'` && echo "...mdim=$mdim"
fdim=`feat-to-dim "scp:$sdata/1/feats.scp" -` && echo "...fdim=$fdim"
idim=0
if [ ! -z "$ivector" ]; then
    idim=`feat-to-dim "ark,t:$ivector" -` && echo "...idim=$idim"
fi
pycmd="n = $mdim - $idim - $fdim; m = 2 * $fdim"
pycmd="$pycmd; assert n % m == 0; print n / m;"
context=`python -c "$pycmd"` || exit 1
echo "...context=$context (auto detected)"

splice_feats="splice-feats --left-context=$context --right-context=$context $feats ark:-"
if [ ! -z "$ivector" ]; then
    splice_feats="$splice_feats | append-vector-to-feats ark:- ark,t:$ivector ark:-"
fi
nnet_prefix=
[ ! -z $device ] && nnet_prefix="CUDA_VISIBLE_DEVICES=$device"

echo "Aligning data in $data using nnet model from $model, outputting to $output_dir"

tra="ark:$UTILS_PATH/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|"
tra_cmd="compile-train-graphs $model/tree $model/final.mdl $lang/L.fst \"$tra\" ark:-"
if [ "$compress" == "true" ]; then
    ali_output="ark,t:|gzip -c >$output_dir/ali.JOB.gz"
else
    ali_output="ark,t:$output_dir/ali.JOB.txt"
fi
$cmd JOB=1:$nj $logs_dir/align-nnet.JOB.log $splice_feats \| \
    $nnet_prefix nnet-forward --use-gpu=$use_gpu --apply-log=true \
        --class-frame-counts=$class_frame_counts $nnet ark:- ark:- \| \
    align-compiled-mapped $scale_opts \
        --beam=$beam --retry-beam=$retry_beam --careful=$careful \
        $model/final.mdl "ark:$tra_cmd |" ark:- "$ali_output" || exit 1;

echo "Done aligning data"
