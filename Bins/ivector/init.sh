#!/bin/bash

function error {
    echo "*** ERROR: $* ***" >&2
    exit 1
}

if [ -z "$KD_ROOT" ]; then
    error "Please set KD_ROOT in ~/.bashrc, e.g. KD_ROOT=/home/public/kaldi/src"
fi

UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"
RUN_CMD=$UTILS_PATH/run.pl
PARSE_OPTS_CMD=$UTILS_PATH/parse_options.sh
WARN_CMD=$UTILS_PATH/summarize_warnings.pl
