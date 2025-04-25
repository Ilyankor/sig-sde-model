import numpy as np
from numpy.typing import NDArray
from itertools import product
 
# initialize N 
C = 2
N = 3

# test2 = np.indices((N,) * N).reshape(N, -1).T + 1
# test2 = np.indices((N,) * N)

# for i in range(1, N+1):
#     yield from product(range(1, C+1), repeat = i)

class Signature:
    """
    A truncated signature.
    """
    def __init__(self, data: NDArray[np.float64], depth: int):
        self.data = np.array(data)
        self.depth = depth
        self.ndim = data.ndim

    def keys(self):
        """
        normal(loc=0.0, scale=1.0, size=None)

        Draw random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, first
        derived by De Moivre and 200 years later by both Gauss and Laplace
        independently [2]_, is often called the bell curve because of
        its characteristic shape (see the example below).

        The normal distributions occurs often in nature.  For example, it
        describes the commonly occurring distribution of samples influenced
        by a large number of tiny, random disturbances, each with its own
        unique distribution [2]_.

        Parameters
        ----------
        loc : float or array_like of floats
            Mean ("centre") of the distribution.
        scale : float or array_like of floats
            Standard deviation (spread or "width") of the distribution. Must be
            non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``loc`` and ``scale`` are both scalars.
            Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized normal distribution.

        See Also
        --------
        scipy.stats.norm : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The probability density for the Gaussian distribution is

        .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                         e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

        where :math:`\\mu` is the mean and :math:`\\sigma` the standard
        deviation. The square of the standard deviation, :math:`\\sigma^2`,
        is called the variance.

        The function has its peak at the mean, and its "spread" increases with
        the standard deviation (the function reaches 0.607 times its maximum at
        :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that
        :meth:`normal` is more likely to return samples lying close to the
        mean, rather than those far away.

        References
        ----------
        .. [1] Wikipedia, "Normal distribution",
               https://en.wikipedia.org/wiki/Normal_distribution
        .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
               Random Variables and Random Signal Principles", 4th ed., 2001,
               pp. 51, 51, 125.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, sigma = 0, 0.1 # mean and standard deviation
        >>> rng = np.random.default_rng()
        >>> s = rng.normal(mu, sigma, 1000)

        Verify the mean and the standard deviation:

        >>> abs(mu - np.mean(s))
        0.0  # may vary

        >>> abs(sigma - np.std(s, ddof=1))
        0.0  # may vary

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, 30, density=True)
        >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        Two-by-four array of samples from the normal distribution with
        mean 3 and standard deviation 2.5:

        >>> rng = np.random.default_rng()
        >>> rng.normal(3, 2.5, size=(2, 4))
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

        """
        for i in range(1, N+1):
            yield from product(range(1, C+1), repeat = i)





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
        data: shape(T, d+1) where T is number of times, d the dimension of data.
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
