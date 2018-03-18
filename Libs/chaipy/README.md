# CHAI Python Libraries

## Prerequisite

* [SciPy stack](http://www.scipy.org/install.html) (includes Python and NumPy)
* [scikit-learn](http://scikit-learn.org/stable/install.html)
* [Theano](http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu) (includes GPU configuration)
* [keras](https://keras.io/)

In addition, you'll also need the [pdnn](../pdnn) toolkit. This is already included in `chai_share`. To install, simply add this line to `~/.bashrc`:

```
export PYTHONPATH=$PYTHONPATH:$CHAI_SHARE_PATH/Libs/pdnn
```

For `pdnn`, it is important that you use the version in `chai_share` and NOT the [original version](https://www.cs.cmu.edu/~ymiao/pdnntk.html). I have modified `pdnn` and some features in `chaipy` are dependent on these changes.

## Usage Instructions

Add the following line to `~/.bashrc` or a equivalent file:

```
export PYTHONPATH=$PYTHONPATH:$CHAI_SHARE_PATH/Libs
```

Then run `source ~/.bashrc` from the terminal. After that, modules can be
imported using the namespace `chaipy`, for example:

```
from chaipy.common import metrics
print metrics.UAR(['red', 'green', 'red'], ['red', 'green', 'green'])
```
