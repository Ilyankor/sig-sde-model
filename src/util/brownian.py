import numpy as np
from numpy.typing import NDArray


def correlated_brownian(
        rho: float,
        t0: float,
        tn: float,
        n: int,
        rng: np.random.Generator = np.random.default_rng()
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates 2 correlated 1-D Brownian motions with initial conditions 
    W(t0) = 0, over a uniform grid.

    Args:
        rho: Coefficient of correlation, -1 <= rho <= 1.
        t0: Initial time.
        tn: Final time.
        n: Number of time steps.
        rng: NumPy random number generator.

    Returns:
        t: Grid points.
        dw: Brownian increments corresponding to each motion.
        w: Correlated Brownian motions in a NumPy array as columns.
    """

    t, dt = np.linspace(t0, tn, num=n+1, retstep=True)

    # sample Brownian increments
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, rho], [rho, 1.0]])
    dw = np.sqrt(dt) * rng.multivariate_normal(mean, cov, size=n)

    # construct Brownian motion
    w = np.zeros((n+1, 2))
    w[1:, :] = np.cumsum(dw, axis=0)
 
    return t, dw, w


def estimate_brownian(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Estimates the underlying Brownian motion of time series data using
    quadratic variation, assuming independence of each dimension.

    Args:
        data: Time series data, with shape (n, d) where n is the number of times
        and d is the dimension of the data.

    Returns:
        The estimated Brownian motion of the data.
    
    Note:
        This estimation can be improved, more reading/reasearching can be done.
        For now, it is a simple way to estimate Brownian motion.
    """

    # estimate Brownian increments
    delta = np.diff(data, axis = 0)
    qv = np.sum(delta**2, axis=0)
    dw = delta / np.sqrt(qv)

    # construct Brownian motion
    w = np.zeros_like(data)
    w[1:, :] = np.cumsum(dw, axis=0)

    return w
