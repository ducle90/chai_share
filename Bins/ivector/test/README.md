# i-Vector Extraction Test

This folder contains examples for using the i-vector extraction scripts. The
basic premise is that you have 2 datasets: training and test. You wish to
train a UBM and i-vector extractor on the training set, and use that to extract
i-vectors on both the training and test set. Below are descriptions of the
currently available test scripts.

## run\_spk\_CMVN.sh

This script simply extracts speaker i-vectors using MFCC features z-normalized
at the speaker level.

## run\_gender\_CMN.sh

This script also extracts speaker i-vectors, but MFCC features are mean 
normalized (i.e. CMVN with no variance normalization) at the gender level.
Moreover, normalization is done using statistics computed on the training set.
