#!/bin/bash

########################################################
# Perform forced alignment using Keras acoustic model. #
# Script adapted from Kaldi's WSJ recipe.              #
########################################################

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"
KD_SCRIPTS_DIR="$CHAI_SHARE_PATH/Bins/kaldi"

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
nnet_name="final.lstm"
ivector=
primary_task=
nutts=1
delay=0

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
    echo "  --use-gpu <yes|no>"
    echo "  --device <id>           # GPU to use (doesn't apply if --use-gpu set to 'no')"
    echo "  --nnet-name <str>       # Custom model name (default: final.lstm)"
    echo "  --ivector <str>         # Path to textual ark containing i-vectors"
    echo "                          # to append to features."
    echo "  --primary-task <tid>    # Set to enable multi-task model decoding"
    echo "  --nutts <nutts>         # No. of utts to send to the model at once"
    echo "  --delay <delay>         # Output delay in frames"
    exit 1
fi
data=$1
lang=$2
model=$3
output_dir=$4
logs_dir=$output_dir/logs

nnet=$model/$nnet_name
class_frame_counts=$model/prior_counts

for f in $data/text $lang/oov.int $model/tree $model/final.mdl \
        ${nnet}.json ${nnet}.weights $class_frame_counts; do
    [ ! -f $f ] && echo "Expected file $f to exist" && exit 1
done

oov=`cat $lang/oov.int` || exit 1
mkdir -p $logs_dir
echo $nj >$output_dir/num_jobs
sdata=$data/split${nj}
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || \
    (where=`pwd` && cd $data && data_path=`pwd` && cd $UTILS_PATH/.. && \
        utils/split_data.sh $data_path $nj && cd $where) || exit 1

feats="$sdata/JOB/feats.scp"

if [ "$use_gpu" == "no" ]; then
    flags="device=cpu,floatX=float32,mode=FAST_RUN"
else
    flags="device=gpu$device,floatX=float32,mode=FAST_RUN"
fi

echo "Detect context size from data"
pycmd="from keras.models import model_from_json; from chaipy.io import json_load"
pycmd="$pycmd; m = model_from_json(json_load('${nnet}.json'))"
mdim=`THEANO_FLAGS=device=cpu python -c "$pycmd; print m.input_shape[-1];"` && \
    echo "...mdim=$mdim"
fdim=`feat-to-dim "scp:$sdata/1/feats.scp" -` && echo "...fdim=$fdim"
idim=0
if [ ! -z "$ivector" ]; then
    idim=`feat-to-dim "ark,t:$ivector" -` && echo "...idim=$idim"
fi
pycmd="n = $mdim - $idim - $fdim; m = 2 * $fdim"
pycmd="$pycmd; assert n % m == 0; print n / m;"
context=`python -c "$pycmd"` || exit 1
echo "...context=$context (auto detect)"

ivec_str=
if [ ! -z "$ivector" ]; then
    ivec_str="--ivectors $ivector"
fi

mt_str=
if [ ! -z "$primary_task" ]; then
    mt_str="--primary-task $primary_task"
fi

echo "Aligning data in $data using Keras model from $model, outputting to $output_dir"

tra="ark:$UTILS_PATH/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|"
tra_cmd="compile-train-graphs $model/tree $model/final.mdl $lang/L.fst \"$tra\" ark:-"
if [ "$compress" == "true" ]; then
    ali_output="ark,t:|gzip -c >$output_dir/ali.JOB.gz"
else
    ali_output="ark,t:$output_dir/ali.JOB.txt"
fi
$cmd JOB=1:$nj $logs_dir/align-keras.JOB.log \
    THEANO_FLAGS=$flags python $KD_SCRIPTS_DIR/keras_forward.py \
        --context $context --padding replicate $ivec_str $mt_str \
        --nutts $nutts --delay $delay --class-frame-counts $class_frame_counts \
        ${nnet}.json ${nnet}.weights $feats \| \
    align-compiled-mapped $scale_opts \
        --beam=$beam --retry-beam=$retry_beam --careful=$careful \
        $model/final.mdl "ark:$tra_cmd |" ark:- "$ali_output" || exit 1;

echo "Done aligning data"
