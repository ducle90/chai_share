#!/bin/bash

##############################################
# Train flat-start monophone acoustic model. #
# Script adapted from Kaldi's WSJ recipe.    #
##############################################

#----------------- CONSTANTS ----------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"

#----------------- COMMAND-LINE ARGUMENTS ------------------#
nj=4            # Number of parallel jobs
stage=-4        # Allows restarting when something goes wrong
cmd="$UTILS_PATH/run.pl"
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=40    # Number of iterations of training
max_iter_inc=30 # Last iter to increase #Gauss on
totgauss=1000   # Target #Gauss
careful=false
realign_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38"
power=0.25
init_beam=12    # Initial beam width to use
final_beam=20   # Final beam width to use after 1st pass

#----------------- MAIN PROGRAM -------------------#
. $UTILS_PATH/parse_options.sh
if [ $# != 3 ]; then
    echo "Usage: $0 [options] <data-dir> <lang-dir> <output-dir>"
    echo ""
    echo "Main options (for others, see top of script file)"
    echo "  --nj <nj>               # Number of parallel jobs"
    echo "  --totgauss <gauss>      # Target number of Gaussians"
    echo "  --init-beam <beam>      # Initial beam used in first training pass"
    echo "  --final-beam <beam>     # Final beam used in later training passes"
    exit 1
fi
data=$1
lang=$2
output_dir=$3
logs_dir=$output_dir/logs

oov_sym=`cat $lang/oov.int` || exit 1

mkdir -p $logs_dir
echo $nj >$output_dir/num_jobs
sdata=$data/split${nj}
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || \
    (where=`pwd` && cd $data && data_path=`pwd` && cd $UTILS_PATH/.. && \
        utils/split_data.sh $data_path $nj && cd $where) || exit 1

feats="scp:$sdata/JOB/feats.scp"
full_feats="scp:$data/feats.scp"

[ ! -f $lang/phones/sets.int ] && exit 1
shared_phones_opt="--shared-phones=$lang/phones/sets.int"

if [ $stage -le -3 ]; then
    if ! feat_dim=`feat-to-dim "$full_feats" - 2>/dev/null` || [ -z $feat_dim ]; then
        feat-to-dim $full_feats -
        echo "Error getting feature dimension"
        exit 1
    fi
    echo "Initialize monophone model"
    $cmd $logs_dir/gmm-init-mono.log gmm-init-mono \
        $shared_phones_opt "--train-feats=$full_feats" $lang/topo $feat_dim \
        $output_dir/0.mdl $output_dir/tree || exit 1
fi

numgauss=`gmm-info --print-args=false $output_dir/0.mdl | grep gaussians | awk '{print $NF}'`
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # Per-iter increment for #Gauss

if [ $stage -le -2 ]; then
    echo "Compiling training graphs"
    $cmd JOB=1:$nj $logs_dir/compile-train-graphs.JOB.log compile-train-graphs \
        $output_dir/tree $output_dir/0.mdl $lang/L.fst \
        "ark:$UTILS_PATH/sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
        "ark:|gzip -c >$output_dir/fsts.JOB.gz" || exit 1
fi

if [ $stage -le -1 ]; then
    echo "Aligning data equally (pass 0)"
    $cmd JOB=1:$nj $logs_dir/align.0.JOB.log align-equal-compiled \
        "ark:gunzip -c $output_dir/fsts.JOB.gz|" "$feats" ark,t:- \| \
        gmm-acc-stats-ali --binary=true $output_dir/0.mdl "$feats" ark:- \
        $output_dir/0.JOB.acc || exit 1
fi

# In the following steps, the --min-gaussian-occupancy=3 option is important,
# o.w. we failt to est. rare phones and later on, they never align properly.

if [ $stage -le 0 ]; then
    echo "Initial re-estimation (pass 0)"
    $cmd $logs_dir/update.0.log gmm-est \
        --min-gaussian-occupancy=3 --mix-up=$numgauss --power=$power \
        $output_dir/0.mdl "gmm-sum-accs - $output_dir/0.*.acc|" $output_dir/1.mdl || exit 1
    rm $output_dir/0.*.acc
fi

beam=$init_beam
x=1
while [ $x -lt $num_iters ]; do
    echo "Pass $x"
    if [ $stage -le $x ]; then
        if echo "$realign_iters" | grep -w $x >/dev/null; then
            echo "...aligning data"
            $cmd JOB=1:$nj $logs_dir/align.${x}.JOB.log gmm-align-compiled \
                $scale_opts --beam=$beam --retry-beam=$[$beam*4] --careful=$careful \
                $output_dir/${x}.mdl "ark:gunzip -c $output_dir/fsts.JOB.gz|" \
                "$feats" "ark,t:|gzip -c >$output_dir/ali.JOB.gz" || exit 1
        fi
        echo "...acc stats"
        $cmd JOB=1:$nj $logs_dir/acc.${x}.JOB.log gmm-acc-stats-ali \
            $output_dir/${x}.mdl "$feats" "ark:gunzip -c $output_dir/ali.JOB.gz|" \
            $output_dir/${x}.JOB.acc || exit 1

        echo "...re-estimating"
        $cmd $logs_dir/update.${x}.log gmm-est \
            --write-occs=$output_dir/$[$x+1].occs --mix-up=$numgauss --power=$power \
            $output_dir/${x}.mdl "gmm-sum-accs - $output_dir/${x}.*.acc|" \
            $output_dir/$[$x+1].mdl || exit 1
        grep "Overall" $logs_dir/update.${x}.log
        rm $output_dir/${x}.{mdl,occs} $output_dir/${x}.*.acc 2>/dev/null
    fi
    if [ $x -le $max_iter_inc ]; then
        numgauss=$[$numgauss+$incgauss]
        echo "Number of Gaussians increased to $numgauss"
    fi
    beam=$final_beam
    x=$[$x+1]
done

echo "Cleaning up and linking final models"
rm $output_dir/ali.*.gz
rm $output_dir/final.{mdl,occs} 2>/dev/null
ln -s ${x}.mdl $output_dir/final.mdl
ln -s ${x}.occs $output_dir/final.occs

$UTILS_PATH/summarize_warnings.pl $logs_dir
