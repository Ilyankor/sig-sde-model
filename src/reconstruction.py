import numpy as np
import sklearn.linear_model
from src.util import *
import matplotlib.pyplot as plt
import esig
import sklearn
from tqdm import tqdm

rng = np.random.default_rng()

S0 = 1.0
V0 = 0.08
mu = 0.001
kappa = 0.5
theta = 0.15
sigma = 0.25
rho = 0.5
t0 = 0.0
tn = 1.0
n = 1000
heston_params = (S0, V0, t0, tn, n, mu, kappa, theta, sigma, rho)

t, u, w = heston_euler(*heston_params, rng)
s = u[:, 0]
ws_est = est_brownian(s)
wv_est = est_brownian(u[:, 1])


# t, s, v = heston(*heston_params, rng)
stream = np.column_stack((t, ws_est, wv_est))

samples, channels = stream.shape
depth = 3
sig_keys = esig.sigkeys(channels, depth).strip().split(' ')
features = len(sig_keys) - 1
data = np.zeros((samples, features))

for i in range(2, n+2):
    data[i-1, :] = esig.stream2sig(stream[:i, :], depth)[1:]

lasso_reg = 1e-5
model = sklearn.linear_model.Lasso(alpha=lasso_reg, max_iter=10000)
model.fit(data, s)

n_trials = 2
mse = np.zeros(n_trials)
for i in tqdm(range(n_trials)):
    heston_params = (S0, V0, t0, tn, n, mu, kappa, theta, sigma, rho)
    # t, u, w = heston_euler(*heston_params, rng)
    # s = u[:, 0]
    # ws_est = est_brownian(s)
    # wv_est = est_brownian(u[:, 1])
    t, s, v = heston_qe(*heston_params, rng)
    ws_est = est_brownian(s)
    wv_est = est_brownian(v)

    # t, s, v = heston(*heston_params, rng)
    stream = np.column_stack((t, ws_est, wv_est))

    samples, channels = stream.shape
    depth = 3
    sig_keys = esig.sigkeys(channels, depth).strip().split(' ')
    features = len(sig_keys) - 1
    data = np.zeros((samples, features))

    for j in range(2, n+2):
        data[j-1, :] = esig.stream2sig(stream[:j, :], depth)[1:]

    lasso_reg = 1e-5
    check = model.predict(data)
    mse[i] = sklearn.metrics.mean_squared_error(s, check)

    # plt.plot(s)
    # plt.plot(check)
    # plt.show()

print(np.max(mse))
plt.plot(s)
plt.plot(check)
plt.show()
