#!/bin/bash

#####################################################################
# Retrain a GMM acoustic model using existing tree and alignments.  #
#####################################################################

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"

#----------------- COMMAND-LINE ARGUMENTS ------------------#
stage=-4        # Allows restarting when something goes wrong
cmd="$UTILS_PATH/run.pl"
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=5     # Number of retraining iterations
numgauss=10000  # Target #Gaussians
beam=20
retry_beam=80
careful=false
realign_iters=
power=0.25

#----------------- MAIN PROGRAM -------------------#
. $UTILS_PATH/parse_options.sh
if [ $# != 5 ]; then
    echo "Usage: $0 [options] <data-dir> <lang-dir> <model-dir> <align-dir> <output-dir>"
    echo ""
    echo "Main options (for others, see top of script file)"
    echo "  --numgauss <gauss>      # Target number of Gaussians"
    echo "  --beam <beam>           # Beam width to use"
    echo "  --retry-beam <beam>     # Beam width for retrying"
    exit 1
fi
data=$1
lang=$2
model=$3
align=$4
output_dir=$5
logs_dir=$output_dir/logs

for f in $model/final.mdl $align/ali.1.gz $data/feats.scp $lang/phones.txt \
        $model/tree $model/questions.int $model/questions.qst; do
    [ ! -f $f ] && echo "No such file $f" && exit 1
done

oov_sym=`cat $lang/oov.int` || exit 1
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1
nj=`cat $align/num_jobs` || exit 1
echo "Automatically detected number of jobs: $nj"
mkdir -p $logs_dir
echo $nj >$output_dir/num_jobs

sdata=$data/split${nj}
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || \
    (where=`pwd` && cd $data && data_path=`pwd` && cd $UTILS_PATH/.. && \
        utils/split_data.sh $data_path $nj && cd $where) || exit 1

feats="scp:$sdata/JOB/feats.scp"

echo "Accumulating tree stats"
$cmd JOB=1:$nj $logs_dir/acc_tree.JOB.log acc-tree-stats \
    $model/final.mdl "$feats" \
    "ark:gunzip -c $align/ali.JOB.gz|" $output_dir/JOB.treeacc || exit 1
$cmd $logs_dir/sum_tree_acc.log sum-tree-stats \
    $output_dir/treeacc $output_dir/*.treeacc || exit 1
rm $output_dir/*.treeacc

echo "Copy existing tree"
cp $model/{questions.int,questions.qst,tree} $output_dir

echo "Initialize triphone model"
$cmd $logs_dir/init_model.log gmm-init-model \
    --write-occs=$output_dir/1.occs $output_dir/tree $output_dir/treeacc \
    $lang/topo $output_dir/1.mdl || exit 1
if grep 'no stats' $logs_dir/init_model.log; then
    echo "** The warnings above about 'no stats' generally mean you have **"
    echo "** phones (or groups of phones) in your phone set that had no  **"
    echo "** corresponding data. You should probably figure out whether  **"
    echo "** something went wrong, or whether your data just doesn't     **"
    echo "** happen to have examples of those phones.                    **"
fi

echo "Mixing up Gaussians to $numgauss"
$cmd $logs_dir/mixup.log gmm-mixup \
    --mix-up=$numgauss $output_dir/1.mdl $output_dir/1.occs $output_dir/1.mdl || exit 1
rm $output_dir/treeacc

echo "Converting alignments from $align to use current tree"
$cmd JOB=1:$nj $logs_dir/convert.JOB.log convert-ali \
    $model/final.mdl $output_dir/1.mdl $output_dir/tree \
    "ark:gunzip -c $align/ali.JOB.gz|" "ark:|gzip -c >$output_dir/ali.JOB.gz" || exit 1

echo "Compiling graphs of transcripts"
$cmd JOB=1:$nj $logs_dir/compile_graphs.JOB.log compile-train-graphs \
    $output_dir/tree $output_dir/1.mdl $lang/L.fst \
    "ark:$UTILS_PATH/sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
    "ark:|gzip -c >$output_dir/fsts.JOB.gz" || exit 1

x=1
while [ $x -lt $num_iters ]; do
    echo "Training pass $x"
    if [ $stage -le $x ]; then
        if echo "$realign_iters" | grep -w $x >/dev/null; then
            echo "Aligning data"
            $cmd JOB=1:$nj $logs_dir/align.${x}.JOB.log gmm-align-compiled \
                $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful \
                $output_dir/${x}.mdl "ark:gunzip -c $output_dir/fsts.JOB.gz|" \
                "$feats" "ark:|gzip -c >$output_dir/ali.JOB.gz" || exit 1
        fi
        echo "...acc stats"
        $cmd JOB=1:$nj $logs_dir/acc.${x}.JOB.log gmm-acc-stats-ali \
            $output_dir/${x}.mdl "$feats" "ark:gunzip -c $output_dir/ali.JOB.gz|" \
            $output_dir/${x}.JOB.acc || exit 1

        echo "...re-estimating"
        $cmd $logs_dir/update.${x}.log gmm-est \
            --write-occs=$output_dir/$[$x+1].occs --power=$power \
            $output_dir/${x}.mdl "gmm-sum-accs - $output_dir/${x}.*.acc|" \
            $output_dir/$[$x+1].mdl || exit 1
        grep "Overall" $logs_dir/update.${x}.log
        rm $output_dir/${x}.{mdl,occs} $output_dir/${x}.*.acc 2>/dev/null
    fi
    x=$[$x+1]
done

echo "Cleaning up and linking final models"
rm $output_dir/ali.*.gz
rm $output_dir/final.{mdl,occs} 2>/dev/null
ln -s ${x}.mdl $output_dir/final.mdl
ln -s ${x}.occs $output_dir/final.occs

$UTILS_PATH/summarize_warnings.pl $logs_dir
