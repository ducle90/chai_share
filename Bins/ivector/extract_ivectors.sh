#!/bin/bash

. `dirname $0`/init.sh

# Optional parameters (see usage message for description)
nj=4
spk2utt=""
num_gselect=20
min_post=0.025
post_gz=""
posterior_scale=1

echo "$0 $@"                    # Print the command line for logging.
. $PARSE_OPTS_CMD || exit 1;    # Parse optional parameters.

if [ $# != 4 ]; then
    echo "Extract i-vectors."
    echo ""
    echo "Usage: $0 <feats_dir> <dubm_dir> <ie_dir> <output_dir>"
    echo "    <feats_dir> - Feature directory (see prepare_data.sh)"
    echo "    <dubm_dir> - Diagonal UBM directory (see train_dubm.sh)"
    echo "    <ie_dir> - i-vector extractor directory (see train_ivector_extractor.sh)"
    echo "    <output_dir> - Output directory"
    echo ""
    echo "Configurable optional parameters:"
    echo "    --nj [4] - Number of jobs to run in parallel"
    echo "    --spk2utt [] - Mapping from speaker name to utterance names. If not set, extract"
    echo "                   utterance i-vectors instead. You can use the 'JOB' variable inside"
    echo "                   this string, e.g. '/some/dir/JOB/spk2utt'."
    echo "    --num-gselect [20] - Number of Gaussians to retain per frame."
    echo "                         Should match the number used in i-vector extractor training."
    echo "    --min-post [0.025] - Minimum posterior to use (posteriors below this are pruned out)."
    echo "                         Should match the number used in i-vector extractor training."
    echo "    --post-gz [] - Pre-computed posterior results. This can be used to speed up i-vector"
    echo "                   extraction. If not set, compute posteriors automatically. You can use"
    echo "                   the 'JOB' variable inside this string, e.g. '/some/dir/post.JOB.gz'."
    echo "                   If this is set, posterior-scale will not be applied. We assume that"
    echo "                   these pre-computed posteriors have already been scaled appropriately."
    echo "    --posterior-scale [1] - Scale on acoustic posteriors, intended to account for inter-frame"
    echo "                            correlation. This is the same as scaling up i-vector priors,"
    echo "                            which tend to produce smaller i-vectors where data-counts are"
    echo "                            small. It's not so important that this matches the value used"
    echo "                            when training the i-vector extractor. Note that this does not"
    echo "                            have any effect if --post-gz is specified."
    echo ""
    echo "The following items will be written to the output directory:"
    echo "    ivectors.ark - Extracted i-vectors"
    echo "    post.*.gz - (only if post-gz is not set) Posterior computation results, can be"
    echo "                used to speed up subsequent i-vector extraction on this dataset."
    echo "    (All log files will be stored in the logs directory)"
    exit 1
fi
feats_dir=$1
dubm_dir=$2
ie_dir=$3
output_dir=$4
log_dir=$output_dir/logs

for f in feats.scp utt2spk spk2utt; do
    [ ! -f "$feats_dir/$f" ] && echo "$0: expecting file $feats_dir/$f to exist" && exit 1
done
for f in final.dubm; do
    [ ! -f "$dubm_dir/$f" ] && echo "$0: expecting file $dubm_dir/$f to exist" && exit 1
done
for f in final.ie; do
    [ ! -f "$ie_dir/$f" ] && echo "$0: expecting file $ie_dir/$f to exist" && exit 1
done
sdata=$feats_dir/split$nj
[[ ! -d $sdata && $feats_dir/feats.scp -ot $sdata ]] || \
    (where=`pwd` && cd $feats_dir && feats_path=`pwd` && cd $UTILS_PATH/.. && \
    ./utils/split_data.sh $feats_path $nj && cd $where) || exit 1
feats="scp:$sdata/JOB/feats.scp"
dubm="$dubm_dir/final.dubm"
ivector_extractor="$ie_dir/final.ie"
mkdir -p $log_dir

if [ -z $post_gz ]; then
    echo "Do Gaussian selection and posterior computation"
    $RUN_CMD JOB=1:$nj $log_dir/gmm-get-post.JOB.log gmm-global-get-post \
        --n=$num_gselect --min-post=$min_post $dubm "$feats" ark:- \| \
        scale-post ark:- $posterior_scale "ark:|gzip -c >$output_dir/post.JOB.gz" || exit 1;
    post_gz=$output_dir/post.JOB.gz
fi

echo "Extract i-vectors"
if [ -z $spk2utt ]; then
    SPK2UTT=""
else
    SPK2UTT="--spk2utt=ark,t:$spk2utt"
fi
$RUN_CMD JOB=1:$nj $log_dir/ivector-extract.JOB.log ivector-extract \
    --compute-objf-change=true $SPK2UTT \
    $ivector_extractor "$feats" "ark:gunzip -c $post_gz|" \
    ark,t:$output_dir/ivectors.JOB.ark || exit 1;
cat $output_dir/ivectors.*.ark >$output_dir/ivectors.ark
rm $output_dir/ivectors.*.ark

echo "*** Finished! Results are in $output_dir"
$WARN_CMD $log_dir
