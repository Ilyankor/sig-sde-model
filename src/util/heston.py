import numpy as np
from numpy.typing import NDArray
from src.util.brownian import correlated_brownian


def heston(
        u: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
        mu: float,
        kappa: float,
        theta: float,
        sigma: float,
    ) -> NDArray[np.float64]:
    """
    Computes the right hand side of the Heston model.
    """
    
    s = u[0]
    v = np.maximum(u[1], 0.0) # ensure v >= 0

    return np.array([
        mu * s * dt + s * np.sqrt(v) * dw[0],
        kappa * (theta - v) * dt + sigma * np.sqrt(v) * dw[1]
    ])


def heston_euler(
        s0: float,
        v0: float,
        t0: float,
        tn: float,
        n: int,
        mu: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        rng: np.random.Generator = np.random.default_rng()
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulates price and volatility on a uniform grid based on the Heston model
    using the Euler-Maruyama method.

    Args:
        s0: Initial price.
        v0: Initial variance.
        t0: Initial time.
        tn: Final time.
        n: Number of time steps.
        mu: Constant drift term.
        kappa: Mean reversion rate.
        theta: Long run variance.
        sigma: Volatility of the variance.
        rho: Correlation coefficient for Brownian motions, -1 <= rho <= 1.
        rng: NumPy random Generator.

    Returns:
        t: Grid points.
        u: NumPy array of simulated price and volatility, [price, volatility].
        w: The underlying Brownian motion used to generate the paths.
    """

    # initialize paths
    u = np.zeros((n+1, 2))
    u[0, 0], u[0, 1] = s0, v0

    # simulate Brownian motion
    t, dw, w = correlated_brownian(rho, t0, tn, n, rng)
    dt = t[1] - t[0]

    # Euler-Maruyama
    for i in range(n):
        u[i+1, :] = u[i] + heston(u[i], dt, dw[i], mu, kappa, theta, sigma)
    
    return t, u, w


def heston_qe(
        s0: float,
        v0: float,
        t0: float,
        tn: float,
        n: int,
        mu: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        rng: np.random.Generator = np.random.default_rng(),
        psi_crit: float = 1.5,
        gamma_1: float = 0.5,
        gamma_2: float = 0.5,
        martingale: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:

    """
    Simulates price and volatility on a uniform grid based on the Heston model
    using the quadratic exponential discretization scheme.

    Args:
        s0: Initial price.
        v0: Initial variance.
        mu: Constant drift term.
        kappa: Mean reversion rate.
        theta: Long run variance.
        sigma: Volatility of the variance.
        rho: Correlation coefficient for Brownian motions, -1 <= rho <= 1.
        t0: Initial time.
        tn: Final time.
        n: Number of time steps.
        rng: NumPy random Generator.
        psi_crit: Matching quadratic or exponential scheme, 1 <= psi_crit <= 2.
        gamma_1: Parameter for algorithm, 0 <= gamma_1 <= 1.
        gamma_2: Parameter for algorithm, 0 <= gamma_2 <= 1.
        martingale: Whether or not to enforce local martingale condition.
    
    Returns:
        t: Grid of times.
        u: NumPy array of simulated price and volatility, [price, volatility].
    """

    # initialize
    t, dt = np.linspace(t0, tn, num=n+1, retstep=True)

    S = np.zeros(n+1)
    S[0] = s0

    V = np.zeros(n+1)
    V[0] = v0

    # constants
    ekt = np.exp(-kappa * dt)
    krs = kappa * rho / sigma

    k0 = -krs * theta * dt
    k1 = gamma_1 * dt * (krs - 0.5) - (rho / sigma)
    k2 = gamma_2 * dt * (krs - 0.5) + (rho / sigma)
    k3 = gamma_1 * dt * np.sqrt(1.0 - rho**2)
    k4 = gamma_2 * dt * np.sqrt(1.0 - rho**2)

    A = k2 + 0.5 * k4
    
    for i in range(n):
        s = S[i]
        v = V[i]

        m = theta + (v - theta) * ekt
        s2 = (sigma**2 / kappa) * (v * ekt * (1.0 - ekt) + 0.5 * theta * (1.0 - ekt)**2)
        psi = s2 / m**2

        # moment matching exponential scheme
        if psi <= psi_crit:
            b2 = 2.0 / psi - 1.0 + 2.0 * np.sqrt(2.0 / psi) * np.sqrt(2.0 / psi - 1.0)
            a = m / (1.0 + b2)

            z = rng.standard_normal()
            v = a * (np.sqrt(b2) + z)**2
            V[i+1] = v

            if martingale:
                k0 = -A * b2 * a / (1.0 - 2.0 * A * a) + 0.5 * np.log(1 - 2.0 * A * a) - (k1 + 0.5 * k3) * v

        # moment matching qudratic scheme
        else:
            p = (psi - 1.0) / (psi + 1.0)
            beta = (1.0 - p) / m

            u = rng.random()
            if u <= p:
                v = 0
            else:
                v = np.log((1.0 - p) / (1.0 - u)) / beta
            V[i+1] = v
            
            if martingale:
                k0 = -np.log(p + beta * (1.0 - p) / (beta - A)) - (k1 + 0.5 * k3) * v

        z = rng.standard_normal()
        logs = np.log(s) + mu * dt + k0 + k1 * V[i] + k2 * v + np.sqrt(k3 * V[i] + k4 * v) * z
        S[i+1] = np.exp(logs)
    
    return t, np.column_stack((S, V))
