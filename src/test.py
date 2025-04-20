import numpy as np
from signature import signature

seq = np.array([
    [1.0, 1.0],
    [2.0, 4.0],
    [6.0, 3.0],
    [8.0, 7.0]
])
depth = 4

result = signature(seq, depth)
print(result)
