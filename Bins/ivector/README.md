# i-Vector Extraction

i-vectors are commonly used features for speaker verification as well as speech recognition (as additional features). These scripts provide an easy-to-use recipe for training and extracting i-vectors.

At a high-level, the extraction process happens in this order:

1. `prepare_data.sh` - Convert WAV files to MFCCs, z-normalize, and perform VAD (optional)
2. `train_dubm.sh` - Train diagonal UBM (Universal Background Model)
3. `train_ivector_extractor.sh` - Train i-vector extractor
4. `extract_ivectors.sh` - Extract i-vectors

## Prerequisite

### Kaldi

Follow the instructions here to install Kaldi:

<https://github.com/kaldi-asr/kaldi/>

Then add these lines to your `~/.bashrc`:

```                                                                             
export KD_ROOT=/path/to/kaldi/src
export PATH=$PATH:$KD_ROOT/bin:$KD_ROOT/featbin:$KD_ROOT/fgmmbin
export PATH=$PATH:$KD_ROOT/fstbin:$KD_ROOT/gmmbin:$KD_ROOT/ivectorbin
export PATH=$PATH:$KD_ROOT/kwsbin:$KD_ROOT/latbin:$KD_ROOT/lmbin
export PATH=$PATH:$KD_ROOT/nnet2bin:$KD_ROOT/nnet3bin:$KD_ROOT/nnetbin
export PATH=$PATH:$KD_ROOT/online2bin:$KD_ROOT/onlinebin
export PATH=$PATH:$KD_ROOT/sgmm2bin:$KD_ROOT/sgmmbin
export PATH=$PATH:$KD_ROOT/../tools/openfst/bin
```

From the terminal, run:

```
source ~/.bashrc
ivector-extract-online2
```

### Dataset Preparation

For each dataset, you will also need to prepare:

* A mapping from utterance name to WAV file (`wav.scp`)
* A mapping from utterance name to speaker name (`utt2spk`)
* A mapping from speaker name to utterance names (`spk2utt`)

Note that:

1. `utt2spk` and `spk2utt` can be used to group utterances into broader groups, such as gender, age, diagnosis, etc...
2. `utt2spk` can be converted to `spk2utt` by running:
    
    ```
    $KD_ROOT/../egs/wsj/s5/utils/utt2spk_to_spk2utt.pl utt2spk >spk2utt
    ```

3. There's also a very similar script to convert `spk2utt` to `utt2spk`:

    ```
    $KD_ROOT/../egs/wsj/s5/utils/spk2utt_to_utt2spk.pl spk2utt >utt2spk
    ```

## Usage

Each of these scripts has a number of optional and required parameters. The best way to learn about them is to look at the scripts themselves. If you don't want to read the entire script, the top of each file provides a summary.

In general, a script invocation looks something like this:

```
./train_dubm.sh --num-iters 10 /feats/dir 128 /output/dir
```

Here, `--num-iters` is an optional parameter and `10` is its value. The rest are required parameters. You can also call the script without any argument to print the usage message.

Examples of how these scripts are used can be found in the [test](Bins/ivector/test) folder. It is highly recommended that you study the examples closely.

## Tips and Troubleshooting

### Audio Sampling Rate

Make sure that your WAV files match the sample frequency in `prepare_data.sh` (default to 16kHz). You can use tools like [sox](http://sox.sourceforge.net/) to resample the audio files.

### Normalization

Play around with the normalization methods! Usually speaker z-normalization works well, but this may change depending on your application. How normalization is performed is controlled through `utt2spk` and `spk2utt`. Some examples can be found in the [test](Bins/ivector/test/) folder.

### Voice Activity Detection

Kaldi's VAD uses a very simple energy-based method that doesn't work well on noisy data. If you can, first do VAD using more sophisticated methods to get new WAV files, then invoke `prepare_data.sh` with `--do-vad false`.
