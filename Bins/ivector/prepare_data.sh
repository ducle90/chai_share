#!/bin/bash

. `dirname $0`/init.sh

# Optional parameters (see usage message for description)
sample_freq=16000
dither=1.0
num_ceps=13
norm_vars="true"
delta_order=2
do_vad="true"
cmvn_scp=""

echo "$0 $@"                    # Print the command line for logging
. $PARSE_OPTS_CMD || exit 1;    # Parse optional parameters

if [ $# != 1 ]; then
    echo "Perform feature extraction and normalization on a set of WAV files."
    echo ""
    echo "Usage: $0 <feats_dir>"
    echo "    <feats_dir> - Feature directory which must contain the following:"
    echo "        wav.scp: Mapping from utterance name to wav files"
    echo "        utt2spk - Mapping from utterance name to speaker name"
    echo "        spk2utt - Mapping from speaker name to utterance names"
    echo ""
    echo "Configurable optional parameters:"
    echo "    --sample-freq [16000] - Sample frequency of audio (must match source audio)"
    echo "    --dither [1.0] - Amount of dithering to use (set to 0 for no dithering)"
    echo "    --num-ceps [13] - Number of cepstral coefficients to output"
    echo "    --norm-vars (true|false) [true] - Whether or not to perform variance normalization"
    echo "    --delta-order [2] - Number of delta orders to use"
    echo "    --do-vad (true|false) [true] - Whether or not to perform VAD as a preprocessing step"
    echo "    --cmvn-scp [] - CMVN stats to use for normalization. This can be used to normalize"
    echo "                    the current dataset using statistics from another dataset. If not set,"
    echo "                    use the dataset's own CMVN stats for normalization."
    echo ""
    echo "The following items will be written to the feature directory:"
    echo "    feats.scp and feats.ark - Final features"
    echo "    cmvn.scp and cmvn.ark - CMVN stats computed on this dataset"
    echo "    vad.ark - Result of VAD (only if do_vad=true)"
    echo "    (All log files will be stored in the logs directory)"
    exit 1
fi
feats_dir=$1
convert_list="$feats_dir/wav.scp"
utt2spk="$feats_dir/utt2spk"
spk2utt="$feats_dir/spk2utt"
output_dir="$feats_dir"
log_dir="$output_dir/logs"

for f in $convert_list $utt2spk $spk2utt; do
    [ ! -f "$f" ] && echo "$0: expecting file $f to exist" && exit 1
done
mkdir -p $log_dir

echo "Extract raw MFCC (with energy by default)"
$RUN_CMD $log_dir/compute-mfcc-feats.log compute-mfcc-feats \
    --sample-frequency=$sample_freq --dither=$dither --num-ceps=$num_ceps \
    scp,p:$convert_list ark,scp:$output_dir/feats_raw.ark,$output_dir/feats_raw.scp || exit 1;
suffix="raw"

if [ "$do_vad" == "true" ]; then
    echo "Do VAD"
    $RUN_CMD $log_dir/compute-vad.log compute-vad \
        scp:$output_dir/feats_raw.scp ark,t:$output_dir/vad.ark || exit 1;
    grep "Applied\|Proportion" $log_dir/compute-vad.log
    echo "Select voiced frames"
    $RUN_CMD $log_dir/select-voiced-frames.log select-voiced-frames \
        scp:$output_dir/feats_raw.scp ark,t:$output_dir/vad.ark \
        ark,scp:$output_dir/feats_voiced.ark,$output_dir/feats_voiced.scp || exit 1;
    grep "Done" $log_dir/select-voiced-frames.log
    suffix="voiced"
fi

echo "Compute CMVN stats"
$RUN_CMD $log_dir/compute-cmvn-stats.log compute-cmvn-stats \
    --spk2utt=ark,t:$spk2utt scp:$output_dir/feats_${suffix}.scp \
    ark,scp:$output_dir/cmvn.ark,$output_dir/cmvn.scp || exit 1;

echo "Apply CMVN"
if [ -z $cmvn_scp ]; then
    cmvn_scp=$output_dir/cmvn.scp
fi
$RUN_CMD $log_dir/apply-cmvn.log apply-cmvn \
    --norm-vars=$norm_vars --utt2spk=ark,t:$utt2spk \
    scp:$cmvn_scp scp:$output_dir/feats_${suffix}.scp \
    ark,scp:$output_dir/feats_cmvn.ark,$output_dir/feats_cmvn.scp || exit 1;
    
echo "Add deltas"
$RUN_CMD $log_dir/add-deltas.log add-deltas \
    --delta-order=$delta_order scp:$output_dir/feats_cmvn.scp \
    ark,scp:$output_dir/feats.ark,$output_dir/feats.scp || exit 1;

echo "Clean up"
rm $output_dir/feats_*.{ark,scp}

echo "*** Finished! Results are in $output_dir"
$WARN_CMD $log_dir
