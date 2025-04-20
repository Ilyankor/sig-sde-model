import numpy as np
from numpy.typing import NDArray


def exp_map(vec: NDArray[np.float64], n: int) -> list:
    """
    Computes the truncated exponential of a vector v.

    Args:
        vec: The vector, shape (1, d).
        n: Depth of truncation.
    
    Returns:
        A list of tensors of rank 0 to n.
    """

    result = [None] * (n+1)
    result[0] = 1.0
    result[1] = vec

    factorial = 1.0
    vec_i = vec

    for i in range(2, n+1):
        factorial *= i
        vec_i = np.tensordot(vec_i, vec, axes=0)
        result[i] = vec_i / factorial

    return result


def multiply(a: list, b: list) -> list:
    """
    Multiplies two elements of the truncated tensor algebra.

    Args:
        a: First element.
        b: Second element.

    Returns:
        a * b where * is the multiplication of the tensor algebra.
    """

    n = len(a)
    result = [None] * n
    result[0] = 1.0

    for i in range(1, n):
        a_0i = a[:(i+1)]
        b_0i = b[i::-1]
        c_i = np.zeros_like(a[i])

        for j in range(i+1):
            c_i += np.tensordot(a_0i[j], b_0i[j], axes=0)

        result[i] = c_i

    return result


def signature(data: NDArray[np.float64], n: int) -> list:
    """
    Computes the signature of input data.
    
    Args:
        data: shape(T, d) where T is number of times, d-1 the dimension of data.
        n: Depth to compute the signature to.
    
    Returns:
        The signature of the data, a list of tensors of rank 0 to n.
    """

    delta = np.diff(data, axis=0)
    T = delta.shape[0]

    result = exp_map(delta[0, :], n)
    for i in range(T-1):
        result = multiply(result, exp_map(delta[i+1], n))

    return result


if __name__ == "__main__":
    exit(0)
