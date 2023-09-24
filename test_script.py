import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import denoise

# parameters
eps = 0.001
mu = 500.

# Parameters for backtracking line search
alpha = 0.01
beta = 0.5

y = np.loadtxt('test_data.csv')
y = np.concatenate([y for _ in range(3)])
n = len(y)

A = denoise.ATV_denoise()
A._verbose = False
x = A.denoise(y, mu=1e3)

# visualize
plt.close('all')
fig,ax = plt.subplots()
ax.plot(np.arange(n), y, label='input')
ax.plot(np.arange(n), x, label=f'denoised (mu={A._mu:.1f})')
ax.legend(loc=0)

"""
# investigating influence of regularization parameter mu
m = 100
mu_grid = np.linspace(0.,100.,m)
cost_grid = np.zeros(m)
norm2_grid = np.zeros(m)
tv_grid = np.zeros(m)
for i in tqdm(range(m), ascii=True):
    x[:] = A.denoise(y, mu=mu_grid[i])
    cost_grid[i] = A._cost(x, y)
    norm2_grid[i] = (y-x).dot(y-x)
    tv_grid[i] = mu_grid[i]*np.abs(np.diff(x)).sum()

fig,ax = plt.subplots()
cost_bound = y.dot(y) - (y.sum())**2/n # n * np.var(y)
ax.plot(mu_grid, cost_grid, marker='.', label='cost')
ax.plot(mu_grid, norm2_grid, marker='.', label='norm2')
ax.plot(mu_grid, tv_grid, marker='.', label='mu*tv')
ax.hlines(cost_bound,mu_grid[0],mu_grid[-1],label='bound')
ax.legend(loc=0)
"""

plt.show(block=False)
