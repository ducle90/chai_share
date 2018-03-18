#!/bin/bash

###########################################
# Do decoding using GMM acoustic model.   #
# Script adapted from Kaldi's WSJ recipe. #
###########################################

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"

#----------------- COMMAND-LINE ARGUMENTS ------------------#
nj=4            # Number of parallel jobs
num_threads=1   # If > 1, will use gmm-latgen-faster-parallel
cmd="$UTILS_PATH/run.pl"
max_active=7000
beam=13
lattice_beam=6
acwt=0.083333
lat_name_mod=

#----------------- MAIN PROGRAM -------------------#
. $UTILS_PATH/parse_options.sh
if [ $# != 3 ]; then
    echo "Usage: $0 [options] <graph-dir> <data-dir> <output-dir>"
    echo "  ...where <output-dir> is assumed to be a subdir of model directory"
    echo ""
    echo "Main options (for others, see top of script file)"
    echo "  --nj <nj>               # Number of parallel jobs"
    echo "  --num-threads <threads> # Number of parallel threads"
    echo "  --acwt <float>          # Acoustic scale used for lattice generation"
    echo "  --lat-name-mod <str>    # Append this string to lattice file name"
    exit 1
fi
graph=$1
data=$2
output_dir=$3
logs_dir=$output_dir/logs

mkdir -p $logs_dir
echo $nj >$output_dir/num_jobs
sdata=$data/split${nj}
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || \
    (where=`pwd` && cd $data && data_path=`pwd` && cd $UTILS_PATH/.. && \
        utils/split_data.sh $data_path $nj && cd $where) || exit 1

model=`dirname $output_dir`/final.mdl   # Model dir is 1 level up from output dir
feats="scp:$sdata/JOB/feats.scp"

for f in $model $graph/HCLG.fst $graph/num_pdfs; do
    [ ! -f $f ] && echo "No such file: $f" && exit 1
done

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

num_pdfs=`cat $graph/num_pdfs`
model_pdfs=`am-info --print-args=false $model | grep pdfs | awk '{print $NF}'`
if [ ! "$num_pdfs" -eq "$model_pdfs" ]; then
    echo "Mismatch in num_pdfs ($num_pdfs) with model_pdfs ($model_pdfs)"
    exit 1
fi

[ ! -z "$lat_name_mod" ] && lat_name_mod=".$lat_name_mod"

echo "Begin decoding"
$cmd JOB=1:$nj $logs_dir/decode.JOB.log gmm-latgen-faster$thread_string \
    --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt --allow-partial=true \
    --word-symbol-table=$graph/words.txt $model $graph/HCLG.fst \
    "$feats" "ark:|gzip -c >$output_dir/lat${lat_name_mod}.JOB.gz" || exit 1
