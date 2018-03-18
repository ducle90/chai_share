#!/bin/bash

##########################################################
# Do decoding using DNN acoustic model (in pdnn format). #
# Script adapted from Kaldi's WSJ recipe.                #
##########################################################

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"
KD_SCRIPTS_DIR="$CHAI_SHARE_PATH/Bins/kaldi"

#----------------- COMMAND-LINE ARGUMENTS ------------------#
nj=4            # Number of parallel jobs
num_threads=1   # If > 1, will use latgen-faster-parallel
cmd="$UTILS_PATH/run.pl"
max_active=7000
min_active=200
beam=13
lattice_beam=8
acwt=0.1
use_gpu="no"
device=
nnet_name="final.dnn"
ivector=
chunk_size="1.2g"
lat_name_mod=

#----------------- MAIN PROGRAM -------------------#
. $UTILS_PATH/parse_options.sh
if [ $# != 3 ]; then
    echo "Decode using pdnn acoustic model. Assume that the model outputs "
    echo "normalized probabilities over HMM states. We currently try to "
    echo "detect context size automatically based on the feature dimension "
    echo "and the model input dimension. We'll allow manual specification "
    echo "in the future if the need arises."
    echo ""
    echo "Usage: $0 [options] <graph-dir> <data-dir> <output-dir>"
    echo "  ...where <output-dir> is assumed to be a subdir of model directory"
    echo ""
    echo "Main options (for others, see top of script file)"
    echo "  --nj <nj>               # Number of parallel jobs"
    echo "  --num-threads <threads> # Number of parallel threads"
    echo "  --acwt <float>          # Acoustic scale used for lattice generation"
    echo "  --use-gpu <yes|no>      # Must run as root to use GPU"
    echo "  --device <id>           # GPU to use (doesn't apply if --use-gpu set to 'no')"
    echo "  --nnet-name <str>       # Custom model name (default: final.dnn)"
    echo "  --ivector <str>         # Path to textual ark containing i-vectors"
    echo "                          # to append to features."
    echo "  --chunk-size <str>      # Total memory allocation for all jobs,"
    echo "                          # will be split evenly among jobs."
    echo "                          # Example: '1.4g', '600m', '500.43m'."
    echo "                          # If using GPU, you should tweak this value"
    echo "                          # to avoid out-of-device-memory error."
    echo "  --lat-name-mod <str>    # Append this string to lattice file name"
    exit 1
fi
graph=$1
data=$2
output_dir=$3
logs_dir=$output_dir/logs

#if [ "$use_gpu" == "yes" ] && [ $EUID -ne 0 ]; then
#    echo "$0: Must run as root to use GPU!"
#    exit 1
#fi

mkdir -p $logs_dir
echo $nj >$output_dir/num_jobs
sdata=$data/split${nj}
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || \
    (where=`pwd` && cd $data && data_path=`pwd` && cd $UTILS_PATH/.. && \
        utils/split_data.sh $data_path $nj && cd $where) || exit 1

# Model dir is 1 level up from output dir
model=`dirname $output_dir`/final.mdl
nnet=`dirname $output_dir`/$nnet_name
feats="$sdata/JOB/feats.scp"
class_frame_counts=`dirname $output_dir`/prior_counts

# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thead_string="-parallel --num-threads=$num_threads"

for f in $model $nnet $graph/HCLG.fst $class_frame_counts; do
    [ ! -f $f ] && echo "No such file: $f" && exit 1
done

if [ "$use_gpu" == "no" ]; then
    flags="device=cpu,floatX=float32,mode=FAST_RUN"
else
    flags="device=gpu$device,floatX=float32,mode=FAST_RUN"
fi

echo "Detect context size from data"
pycmd="from chaipy.learning.pdnn import get_dnn_cfg"
pycmd="$pycmd; print get_dnn_cfg('$nnet').n_ins"
mdim=`THEANO_FLAGS=$flags python -c "$pycmd"` && echo "...mdim=$mdim"
fdim=`feat-to-dim "scp:$sdata/1/feats.scp" -` && echo "...fdim=$fdim"
idim=0
if [ ! -z "$ivector" ]; then
    idim=`feat-to-dim "ark,t:$ivector" -` && echo "...idim=$idim"
fi
pycmd="n = $mdim - $idim - $fdim; m = 2 * $fdim"
pycmd="$pycmd; assert n % m == 0; print n / m;"
context=`python -c "$pycmd"` || exit 1
echo "...context=$context (auto detect)"

echo "Computing per-job chunk size"
# We use nj+1 to leave some buffer for models
pycmd="x = '$chunk_size'; size = float(x[:-1]) / ($nj + 1)"
pycmd="$pycmd; print '{:.3f}{}'.format(size, x[-1])"
job_chunk_size=`python -c "$pycmd"` || exit 1
echo "...job_chunk_size=$job_chunk_size"
chunk_str="--chunk-size $job_chunk_size"

ivec_str=
if [ ! -z "$ivector" ]; then
    ivec_str="--ivectors $ivector"
fi
[ ! -z "$lat_name_mod" ] && lat_name_mod=".$lat_name_mod"

echo "Begin decoding"
$cmd JOB=1:$nj $logs_dir/decode-nnet.JOB.log \
    THEANO_FLAGS=$flags python $KD_SCRIPTS_DIR/nnet_forward.py \
        --context $context --padding replicate $ivec_str $chunk_str \
        --class-frame-counts $class_frame_counts $nnet $feats \| \
    latgen-faster-mapped$thread_string \
        --min-active=$min_active --max-active=$max_active \
        --beam=$beam --lattice-beam=$lattice_beam \
        --acoustic-scale=$acwt --allow-partial=true \
        --word-symbol-table=$graph/words.txt $model $graph/HCLG.fst \
        ark,t:- "ark:|gzip -c >$output_dir/lat${lat_name_mod}.JOB.gz" || exit 1
