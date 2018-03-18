#!/bin/bash

. `dirname $0`/init.sh

# Optional parameters (see usage message for description)
nj=4
num_gselect=20
num_iters=10
min_post=0.025
posterior_scale=1
gaussian_min_count=

echo "$0 $@"                    # Print the command line for logging.
. $PARSE_OPTS_CMD || exit 1;    # Parse optional parameters.

if [ $# != 4 ]; then
    echo "Perform i-vector extractor training."
    echo ""
    echo "Usage: $0 <feats_dir> <dubm_dir> <ivector_dim> <output_dir>"
    echo "    <feats_dir> - Feature directory (see prepare_data.sh)"
    echo "    <dubm_dir> - Diagonal UBM directory (see train_dubm.sh)"
    echo "    <ivector_dim> - Desired i-vector dimension"
    echo "    <output_dir> - Output directory"
    echo ""
    echo "Configurable optional parameters:"
    echo "    --nj [4] - Number of jobs to run in parallel"
    echo "    --num-gselect [20] - Number of Gaussians to retain per frame"
    echo "    --num-iters [10] - Number of EM iterations to perform for i-vector extractor training"
    echo "    --min-post [0.025] - Minimum posterior to use (posteriors below this are pruned out)"
    echo "    --posterior-scale [1] - Scale on acoustic posteriors, intended to account for inter-frame"
    echo "                            correlation. This is the same as scaling up i-vector priors."
    echo "    --gaussian-min-count [] - Minimum total count per Gaussian"
    echo ""
    echo "The following items will be written to the output directory:"
    echo "    final.ie - Final i-vector extractor trained on the specified data"
    echo "    post.*.gz - Posterior computation results, can be used to "
    echo "              speed up i-vector extraction on this dataset"
    echo "    (All log files will be stored in the logs directory)"
    exit 1
fi
feats_dir=$1
dubm_dir=$2
ivector_dim=$3
output_dir=$4
log_dir=$output_dir/logs

for f in feats.scp utt2spk spk2utt; do
    [ ! -f "$feats_dir/$f" ] && echo "$0: expecting file $feats_dir/$f to exist" && exit 1
done
for f in final.dubm; do
    [ ! -f "$dubm_dir/$f" ] && echo "$0: expecting file $dubm_dir/$f to exist" && exit 1
done
sdata=$feats_dir/split$nj
[[ ! -d $sdata && $feats_dir/feats.scp -ot $sdata ]] || \
    (where=`pwd` && cd $feats_dir && feats_path=`pwd` && cd $UTILS_PATH/.. && \
        ./utils/split_data.sh $feats_path $nj && cd $where) || exit 1
feats="scp:$sdata/JOB/feats.scp"
dubm="$dubm_dir/final.dubm"
mkdir -p $log_dir

echo "Initialize i-vector extractor"
$RUN_CMD $log_dir/ie-init.log ivector-extractor-init \
    --ivector-dim=$ivector_dim --use-weights=false \
    "gmm-global-to-fgmm $dubm -|" $output_dir/0.ie || exit 1;
        
echo "Do Gaussian selection and posterior computation"
$RUN_CMD JOB=1:$nj $log_dir/gmm-get-post.JOB.log gmm-global-get-post \
    --n=$num_gselect --min-post=$min_post $dubm "$feats" ark:- \| \
    scale-post ark:- $posterior_scale "ark:|gzip -c >$output_dir/post.JOB.gz" || exit 1;
        
echo "Train i-vector extractor"
for iter in `seq 0 $[$num_iters-1]`; do
    echo "--> Training pass: $iter"
            
    echo "Accumulate stats"
    $RUN_CMD JOB=1:$nj $log_dir/ie-acc_${iter}.JOB.log ivector-extractor-acc-stats \
        $output_dir/${iter}.ie "$feats" "ark:gunzip -c $output_dir/post.JOB.gz|" \
        $output_dir/${iter}.JOB.acc || exit 1;

    echo "Summing stats"
    accs=""
    for j in `seq $nj`; do
        accs+="$output_dir/${iter}.${j}.acc "
    done
    $RUN_CMD $log_dir/ie-sum-accs_${iter}.log ivector-extractor-sum-accs \
        $accs $output_dir/${iter}.acc || exit 1;

    gmc=
    [ ! -z "$gaussian_min_count" ] && gmc="--gaussian-min-count=$gaussian_min_count"
    echo "Re-estimate stats"
    $RUN_CMD $log_dir/ie-est_${iter}.log ivector-extractor-est $gmc \
        $output_dir/${iter}.ie $output_dir/${iter}.acc \
        $output_dir/$[$iter+1].ie || exit 1;
    grep "Overall" $log_dir/ie-est_${iter}.log
    rm $output_dir/${iter}.{ie,acc} $output_dir/${iter}.*.acc
done
        
echo "Copy final i-vector extractor"
mv $output_dir/${num_iters}.ie $output_dir/final.ie

echo "*** Finished! Results are in $output_dir"
$WARN_CMD $log_dir
