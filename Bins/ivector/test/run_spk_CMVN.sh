#!/bin/bash

BIN_DIR=`dirname $0`/..
. $BIN_DIR/init.sh
DATA_PATH=/home/ducle/Data/Samples/ivector
CONFIG_PATH=$DATA_PATH/configs
SETS=("train" "test")
NUM_GAUSS=128
IVECTOR_DIM=10

if [ $# != 1 ]; then
    echo "End-to-end example of i-vector extraction. MFCC features are "
    echo "z-normalized at the speaker level."
    echo ""
    echo "Usage: $0 <output_dir>"
    echo "    <output_dir> - Output directory"
    exit 1
fi
output_dir=$1
mkdir -p $output_dir

FEATS_DIR=$output_dir/data
for SET in "${SETS[@]}"; do
    echo "Preparing data for set ${SET}..."
    mkdir -p $FEATS_DIR/$SET
    cp $CONFIG_PATH/convert.${SET}.txt $FEATS_DIR/$SET/wav.scp
    cp $CONFIG_PATH/utt2spk.${SET}.txt $FEATS_DIR/$SET/utt2spk
    cp $CONFIG_PATH/spk2utt.${SET}.txt $FEATS_DIR/$SET/spk2utt
    $BIN_DIR/prepare_data.sh $FEATS_DIR/$SET || exit 1;
done

DUBM_DIR=$output_dir/dubm/$NUM_GAUSS
echo "Train diagonal UBM on training set using $NUM_GAUSS gaussians..."
$BIN_DIR/train_dubm.sh --nj 4 \
    $FEATS_DIR/train $NUM_GAUSS $DUBM_DIR || exit 1;

IE_DIR=$output_dir/ivector_extractor/${NUM_GAUSS}_${IVECTOR_DIM}
echo "Train i-vector extractor with dimension ${IVECTOR_DIM}..."
$BIN_DIR/train_ivector_extractor.sh --nj 4 \
    $FEATS_DIR/train $DUBM_DIR $IVECTOR_DIM $IE_DIR || exit 1;

IVECS_DIR=$output_dir/ivectors/${NUM_GAUSS}_${IVECTOR_DIM}
for SET in "${SETS[@]}"; do
    if [ "$SET" == "train" ]; then
        # Use pre-computed posterior results
        POST_GZ="--post-gz $IE_DIR/post.JOB.gz"
        NJ=4
    else
        POST_GZ=""
        NJ=2
    fi
    echo "Extract i-vectors for set ${SET}..."
    $BIN_DIR/extract_ivectors.sh \
        --nj $NJ $POST_GZ --spk2utt $FEATS_DIR/$SET/split$NJ/JOB/spk2utt \
        $FEATS_DIR/$SET $DUBM_DIR $IE_DIR $IVECS_DIR/$SET || exit 1;
done
