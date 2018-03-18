# CHAI Java Libraries

## Prerequisite

These libraries depend on [ApacheCommons](https://commons.apache.org/). After installing, add the following line to `~/.bashrc`:

```
export CLASSPATH=$CLASSPATH:/path/to/ApacheCommons/*
```

## Usage Instructions

Add the following line to `~/.bashrc`:

```
export CLASSPATH=$CLASSPATH:.:$CHAI_SHARE_PATH/Libs
```

Then run `source ~/.bashrc` from the terminal. After that, packages can be
imported using the namespace `chaijava`, for example:

```
import chaijava.common.Vibos

public static void main(String[] args) {
    System.out.println("true = " + Vibos.asInt(true));
    System.out.println("false = " + Vibos.asInt(false));
}
```
