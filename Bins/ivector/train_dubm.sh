#!/bin/bash

. `dirname $0`/init.sh

# Optional parameters (see usage message for description)
nj=4
num_gselect=30
num_frames=500000
num_iters_init=20
num_iters=5
init_gauss_frac=2
min_gauss_weight=0.0001

echo "$0 $@"                    # Print the command line for logging
. $PARSE_OPTS_CMD || exit 1;    # Parse optional parameters

if [ $# != 3 ]; then
    echo "Perform diagonal UBM training on a collection of acoustic features."
    echo ""
    echo "Usage: $0 <feats_dir> <num_gauss> <output_dir>"
    echo "    <feats_dir> - Feature directory (see prepare_data.sh)"
    echo "    <num_gauss> - Number of Gaussians to use for training the UBM"
    echo "    <output_dir> - Output directory"
    echo ""
    echo "Configurable optional parameters:"
    echo "    --nj [4] - Number of jobs to execute in parallel"
    echo "    --num-gselect [30] - Number of Gaussians to retain per frame"
    echo "    --num-frames [500000] - Number of frames to keep for initialization"
    echo "    --num-iters-init [20] - Number of EM iterations to perform for initialization"
    echo "    --num-iters [5] - Number of EM iterations to perform for UBM training"
    echo "    --init-gauss-frac [2] - Start with a fraction of the target Gaussians."
    echo "                            For example, 2 means start with 1/2 the target Gaussians."
    echo "    --min-gauss-weight [0.0001] - Minimum Gaussian weight before pruning"
    echo ""
    echo "The following item will be written to the output directory:"
    echo "    final.dubm - Final diagonal UBM trained on the specified data"
    echo "    (All log files will be stored in the logs directory)"
    exit 1
fi
feats_dir=$1
num_gauss=$2
output_dir=$3
log_dir=$output_dir/logs

num_gauss_init=$[$num_gauss / $init_gauss_frac]
for f in feats.scp utt2spk spk2utt; do
    [ ! -f "$feats_dir/$f" ] && echo "$0: expecting file $feats_dir/$f to exist" && exit 1
done
sdata=$feats_dir/split$nj
[[ ! -d $sdata && $feats_dir/feats.scp -ot $sdata ]] || \
    (where=`pwd` && cd $feats_dir && feats_path=`pwd` && cd $UTILS_PATH/.. && \
        ./utils/split_data.sh $feats_path $nj && cd $where) || exit 1
feats="scp:$sdata/JOB/feats.scp"
full_feats="scp:$feats_dir/feats.scp"
mkdir -p $log_dir

echo "Initialize UBM"
$RUN_CMD $log_dir/gmm-init.log gmm-global-init-from-feats \
    --num-frames=$num_frames --min-gaussian-weight=$min_gauss_weight \
    --num-gauss=$num_gauss --num-gauss-init=$num_gauss_init --num-iters=$num_iters_init \
    "$full_feats" $output_dir/0.dubm || exit 1;

echo "Do Gaussian selection"
$RUN_CMD JOB=1:$nj $log_dir/gmm-gselect.JOB.log gmm-gselect \
    --n=$num_gselect $output_dir/0.dubm "$feats" \
    "ark:|gzip -c >$output_dir/gselect.JOB.gz" || exit 1;
        
echo "Train UBM"
for iter in `seq 0 $[$num_iters-1]`; do
    echo "--> Training pass: $iter"

    echo "Accumulate stats"
    $RUN_CMD JOB=1:$nj $log_dir/gmm-acc_${iter}.JOB.log gmm-global-acc-stats \
        "--gselect=ark:gunzip -c $output_dir/gselect.JOB.gz|" \
        $output_dir/${iter}.dubm "$feats" $output_dir/${iter}.JOB.acc || exit 1;
            
    echo "Re-estimate stats"
    if [ $iter -lt $[$num_iters-1] ]; then
        # Don't remove low-count Gaussians till last iter, or 
        # gselect info won't be valid anymore.
        rm_lc_g="false"
    else
        rm_lc_g="true"
    fi
    $RUN_CMD $log_dir/gmm-est_${iter}.log gmm-global-est \
        --remove-low-count-gaussians=$rm_lc_g --min-gaussian-weight=$min_gauss_weight \
        $output_dir/${iter}.dubm "gmm-global-sum-accs - $output_dir/${iter}.*.acc|" \
        $output_dir/$[$iter+1].dubm || exit 1;
    grep "Overall" $log_dir/gmm-est_${iter}.log
    rm $output_dir/${iter}.dubm $output_dir/${iter}.*.acc
done

echo "Copy final UBM and cleanup"
mv $output_dir/${num_iters}.dubm $output_dir/final.dubm
rm $output_dir/gselect.*.gz

echo "*** Finished! Results are in $output_dir"
$WARN_CMD $log_dir
