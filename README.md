# Signature model

## Computing signatures

`misc/signature.py` contains a basic implementation of computing the signature of sequence data.

### Usage

Given an ordered sequence of times `t0 < t1 < ... < tn` and corresponding `d` dimensional data, input the data as a `numpy.NDArray`  with shape `(n, d+1)` along with the desired depth of the signature.
The result is a `list` of tensors with rank `0, 1, ..., n` which form the truncated signature.

### Example

```python
import numpy as np

seq = np.array([
    [1.0, 1.0],
    [2.0, 4.0],
    [6.0, 3.0],
    [8.0, 7.0]
])
depth = 3

sig = signature(seq, depth)
```

## A model

???
