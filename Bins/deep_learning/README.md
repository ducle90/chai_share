# Reusable Deep Learning Scripts

These scripts are distilled from my experience in training lots of deep learning models. Nowadays, with high-level deep learning libraries like Keras, Torch, Theano, or TensorFlow, it's very easy to train a relatively complex deep network. Despite having access to these libraries, I keep encountering a number of problems, specifcally:

* The need for **quickly evaluating a large number of hyperparameter combinations**. Manually selecting hyperparameters simply doesn't cut it in this day and age. For example, suppose I want to see which DNN architecture works best for a particular problem. I can first run the experiment with 1 hidden layer, wait for it to finish, then run it with 2 hidden layers, then 3, 4, and so forth. Instead of this tedious process, I'd much rather have a script that can train all these architectures and automatically choose the best one.
* The need to **avoid repeated work**. This means skipping model architectures that have already been trained previously. For a small number of hyperparameters, it is possible to remember what has been done and what hasn't. However, for a larger hyperparameter set, it becomes almost impossible to keep track using only your memory.
* The need to **resume training in case of unexpected failures**. This might not be important when the dataset is small, but it is for large datasets where one epoch takes hours or days to train. In these cases, it is crucial to be able to resume training from a recent checkpoint instead of starting the process all over again.
* The need for **incremental training** (or transfer learning). One of the more interesting problems in deep learning is to leverage models trained on one dataset to improve performance on another dataset. Existing libraries don't directly account for this application.
* The need for **extensibility**. One of the big challenges in research in general is that there's no one-size-fit-all solution. A model or training strategy that works for one domain might not be applicable to another. For a script to be usable in the long term, users must be able to easily extend it to better accommodate their specific use case.

The scripts here are designed with these requirements in mind. If you've had a need for any of these requirements, it is definitely worth your time to take a look at the scripts.

## Prerequisite

### chai_share Setup

Follow the instructions to setup `chai_share`:

1. [Overall Setup](https://github.com/ducle90/chai_share/)
2. [chaipy](../../Libs/chaipy)

### Data Preparation

The scripts assume that your data is in Kaldi format. Kaldi can be setup by following the instructions [here](../ivector#kaldi).

Kaldi data are time series data and contain three main components:

1. `ark`: binary files containing feature values. These files are not readable with regular text editors.
2. `scp`: text files where each line maps an utterance ID to a particular byte position in the ark file. A line in a scp file might look like this: `train_1 /path/feats.ark:123`. This line tells the system how to extract features for the utterance with ID `train_1`. A scp file may point to several ark files, and multiple scp files can point to the same ark file. A general best practice is to store all features in a big ark file, and split the dataset (e.g., into a training, validation, and test set) by trimming the scp file. **NOTE**: the ark paths inside scp files must be absolute.
3. `utt2label`: text files that map each frame in the time series to a label (can be discrete or continuous). For example, a line in a label file might look like this: `train_1 0 5 2`. This means the first frame in utterance `train_1` has label 0, second frame has label 5, and third frame has label 2.

You can extract features (ark and scp) directly using Kaldi. For example, MFB features can be extracted using `compute-fbank-feats` and MFCC features can be extracted using `compute-mfcc-feats`.

You can also convert features from other formats into Kaldi format using `copy-feats`. The general principle is to print features to `stdout`, and `copy-feats` can read this stream and convert it to Kaldi format. For example, suppose you write a script to convert features for two utterances, `train_1` and `train_2` to Kaldi format. Let's call this script `print_feats`. The script will need to output something like this:

```
train_1 [
  0.1 0.2 0.3
  -0.4 0.5 0.7 ]
train_2 [
  0.6 0.7 0.3
  0.01 1.4 2.0
  -0.3 1.2 1.6 ]
```

This means `train_1` has 2 frames, `train_2` has 3 frames, and the feature dimension is 3. You can then create an `ark` and `scp` file for this data by running:

```
print_feats | copy-feats ark,t:- ark,scp:/path/feats.ark,/path/feats.scp
```

For python, there is a utility function `print_matrix` as part of `chaipy.kaldi` that can be used like this:

```
import numpy as np
from chaipy.kaldi import print_matrix

train_1 = np.loadtxt('train_1.txt')
print_matrix('train_1', train_1)
train_2 = np.loadtxt('train_2.txt')
print_matrix('train_2', train_2)
```

The ark and scp can also be printed to `stdout` using `copy-feats`:

```
copy-feats scp:/path/feats.scp ark,t:-
copy-feats ark:/path/feats.ark ark,t:-
```

or save to a regular text file:

```
copy-feats scp:/path/feats.scp ark,t:/path/feats.txt
copy-feats ark:/path/feats.ark ark,t:/path/feats.txt
```

To read more about Kaldi I/O format, see:

http://kaldi-asr.org/doc/io.html

### Handling Very Large Datasets

By default, the entire content of the scp file is loaded into memory. This allows for very fast data loading and manipulation. This is not an issue for relatively small datasets (i.e., less than 100 hours of speech data, assuming 100 frames/second and 40 features/frame). However, this won't work for very large datasets that can't be fit into memory, such as Fisher which has 2000 hours of speech data.

These scripts are capable of handling such datasets through a simple mechanism. All you have to do is split the original scp file into several smaller files, and create a new scp file that points to these smaller files. For example, this is the content of one such file:

```
/home/ducle/Data/AVEC_2016/train_mfb.1.scp
/home/ducle/Data/AVEC_2016/train_mfb.2.scp
```

This means I'm splitting the training set of AVEC into 2 smaller sets. I can then use this file in place of the regular scp when calling the scripts. The code will make sure that only one of the smaller scp files will be loaded into memory at any given time. AVEC is a toy example, but you can imagine splitting the scp file of Fisher into 20 smaller files, each containing roughly 100 hours of speech.

**NOTE**: For training sets, it is usually good practice to shuffle content of the original scp file prior to splitting.

## High-Level Overview

Here I give a high-level description of the available scripts. I intentionally avoid overly specific documentation because the scripts may change often, and it's almost impossible to keep the documentation up-to-date. In general, the best way to learn about the script is:

1. Through the help text. You can see this by running the script with the `-h` option, e.g. `python finetune_rnn.py -h`.
2. By reading the code itself.

All of the scripts here follow the same high-level design. The datasets are divided into two groups: (1) a training set and (2) a collection of validation sets (one of them is designated for actual validation). The script iterates through each set of hyperparameters, trains the model on the training set, and computes errors on the validation sets. The results are logged and the best model (or all models) will be saved to disk.

### finetune_rnn.py

This script trains a simple RNN-based network. The architecture looks like this:

**Input --> RNN --> RNN --> ... --> RNN --> FullyConnected --> Output**

The script supports both classification and regression. For classification, the loss function to minimize is:

```
L(model, dataset) = CrossEntropy(model.output, dataset.labels)
```

For regression, the loss function to minimize is:

```
L(model, dataset) = MeanSquaredError(model.output, dataset.labels)
```

### adapt_rnn.py

This script trains a RNN-based network (similar architecture as `finetune_rnn.py`), but uses a base model to regularize the network. The regularization weight is called `rho`. For classification, the loss function is:

```
L(model, dataset) = (1 - rho) * CrossEntropy(model.output, dataset.labels) + rho * KL-Divergence(model.output, base_model.output)
```

For regression, the loss function is:

```
L(model, dataset) = (1 - rho) * MeanSquaredError(model.output, dataset.labels) + rho * MeanSquaredError(model.output, base_model.output)
```

### finetune_mtrnn.py

This is similar to `finetune_rnn.py`, but each frame can be associated with multiple labels (i.e., tasks). For example, the network can be trained to output both arousal and valence values, or both continuous and categorical emotion labels. The network architecture looks like this:

**Input --> RNN --> RNN --> ... --> RNN --> FullyConnected --> [Output_1, Output_2, ... ]**

Given task weights `w_t`, the loss function is:

```
L_mt(model, dataset) = 0
for t = 1, ntasks:
    L_mt(model, dataset) += w_t * L(model.output[t], dataset.labels[t])
```

where `L` can be `CrossEntropy` (classification) or `MeanSquaredError` (regression), depending on the individual task.

### adapt_mtrnn.py

This is similar to `adapt_rnn.py`, but applied to multi-task settings. It also uses a base model to regularize the network output. Given the regularization weight `rho` and task weights `w_t`, the loss function is:

```
L_mt(model, dataset) = 0
for t = 1, ntasks:
    rho_t = rho * w_t
    L_mt(model, dataset) += (w_t - rho_t) * L(model.output[t], dataset.labels[t]) + rho_t * L_aux(model.output[t], base_model.output[t])
```

For classification, `L` is `CrossEntropy` and `L_aux` is `KL-Divergence`. For regression, both `L` and `L_aux` are `MeanSquaredError`.

## Example Usage

Here I give some examples of using the script to estimate arousal and/or valence for each acoustic frame in the AVEC2016 dataset.

This also assumes that you have setup `chai_share` in your home folder.

### Single-Task RNN Training

From the terminal, run:

```
DATA="/home/ducle/Data/AVEC_2016"
python ~/chai_share/Bins/deep_learning/finetune_rnn.py \
    $DATA/train_mfb.scp $DATA/utt2arousal /tmp/st_arousal \
    --valid-scp $DATA/dev_mfb.scp $DATA/train_mfb.scp --valid-labels $DATA/utt2arousal $DATA/utt2arousal \
    --task regression --use-var-utt False --nutts 1 \
    --layer-type rnn lstm --layer-size 40 --num-layers 1 2 --bidirectional False \
    --init-lr 0.002 --max-epoch 2 \
    >/tmp/st_arousal.log 2>&1
```

I will now explain what each part of the arguments means.

```
    $DATA/train_mfb.scp $DATA/utt2arousal /tmp/st_arousal
```

This specifies `$DATA/train_mfb.scp` as the training features, `$DATA/utt2arousal` as the target labels, and saves experiment results to `/tmp/st_arousal`.

```
    --valid-scp $DATA/dev_mfb.scp $DATA/train_mfb.scp --valid-labels $DATA/utt2arousal $DATA/utt2arousal
```

This specifies the development and training sets (along with their correct labels) to be monitored in the validation pipeline. By default, the first dataset (`dev_mfb.scp`) is used for actual validation, but results for both sets will be logged.

```
    --task regression --use-var-utt False --nutts 1
```

This configures how the data is handled. It means the target task is regression, and training will be done on utterance at a time.

```
    --layer-type rnn lstm --layer-size 40 --num-layers 1 2 --bidirectional False
```

This configures the hyperparameters to be swept over. Specifically, the set of hyperparameters is the Cartesian product `[rnn, lstm] x [40] x [1, 2] x [False]`. In general, arguments that take more than one value are tunable hyperparameters.

```
    --init-lr 0.002 --max-epoch 2
```

This configures the training process by setting the initial learning rate to 0.002 and trains for a maximum of 2 epochs.

```
    >/tmp/st_arousal.log 2>&1
```

This saves the logs of the experiment to `/tmp/st_arousal.log`. In general, the logs contain important information and should most likely be saved for later review. While the script is running, you can view the progress by opening a new terminal and monitoring the log file:

```
tail -f /tmp/st_arousal.log
```

After the experiment finishes, inside `/tmp/st_arousal` you will find:

1. `{best_model_name}.json`, `{best_model_name}.weights`: These two files define the network architecture and weight values of the best hyperparameter combination. You can reconstruct the trained network using these two files.
2. `final.json`, `final.weights`: These two are symbolic links that point to the two files above.
3. `summary.txt`: Each line in this file follows this format: `<model_name> <train_err> <validation_err_1> <validation_err_2>`. In this case, the second validation set happens to be the training set, but `<validation_err_2>` might have different values than `<train_err>`. This is because the latter is accumulated after training each minibatch, whereas the model weights don't change throughout the computation of the former.

A property of `summary.txt` is that the script uses it to determine which hyperparameter combinations have been run before and skip them. This allows you to do incremental parameter sweep without incurring repeated work.

This example only covers a small subset of functionalities available in this script. As always, consult the help text and the code itself if necessary.

### Resuming Training From A Checkpoint

During training, a temporary directory is created for the current hyperparameter set and the models are saved after every epoch. After training on this hyperparameter set, the directory is removed. However, if the training is interrupted mid-way, the directory remains and you can use it to resume training.

For example, suppose we're running the script above and training was interrupted during the second epoch of the first hyperparameter set, `[rnn, 40, 1, False]`. You will find the following directory:

```
/tmp/st_arousal/rnn_1x40+dropout_0.0+l1_0.0+l2_0.0
```

which contains `1.json` and `1.weights`. These define the model after the first training epoch. To resume training, simply create a `resume.json` file in this directory with the following content:

```
{
    "lrate": 0.002,
    "epoch": 2,
    "weights": "1.weights",
    "decay": false
}
```

This tells the script to resume training by loading the weights and restarting the process at epoch 2 with learning rate 0.002. The last token, `decay`, specifies whether the learning rate has started decaying. This token is optional (set to `false` by default), while the first three tokens are required.

## Extending The Scripts

These scripts are driven by the classes defined in [Libs/chaipy/engine](../../Libs/chaipy/engine).

The base class is `FinetuneRNNEngine` in [Libs/chaipy/engine/rnn.py](../../Libs/chaipy/engine/rnn.py).

All of these classes are importable through `chaipy.engine`, and you can extend it like any other class. Alternatively, you can just copy its content to your folder and modify it from there. The first method is more principled and always preferable, of course.

### General Principle

The high-level flow of the script is defined by:

```
def run(self):
    self.parse_args()
    self.init_data()
    self.pre_execute()
    params_title, params_iter = self.params_iter()
    for params in params_iter:
        params_map = OrderedDict()
        for title, param in zip(params_title, params):
            params_map[title] = param
        self.step(**params_map)
    self.post_execute()
```

I will quickly go over each section and explain what they do.

```
    self.parse_args()
    self.init_data()
    self.pre_execute()
```

These 3 lines setup the command-line arguments, initializes the datasets, and performs preparation steps for the experiments.

```
    params_title, params_iter = self.params_iter()
    for params in params_iter:
        params_map = OrderedDict()
        for title, param in zip(params_title, params):
            params_map[title] = param
        self.step(**params_map)
```

These lines loop through each hyperparameter combination and performs an iteration of training.

```
    self.post_execute()
```

This wraps up the experiment.

You can trace the flow of these functions and determine which parts you need to override in your class. For example, you might want to replace the default network architecture with a different architecture, and/or change how the datasets are initialized, and/or how the results are saved to disk.
