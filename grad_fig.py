import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# plots a point density
def plot_grad_dens(x,y,fname):
   # Calculate the point density
   xy = np.vstack([x,y])
   z = gaussian_kde(xy)(xy)

   fig, ax = plt.subplots()
   ax.scatter(x, y, c=z, s=10, edgecolor='')
   plt.grid()
   # plt.show()
   plt.savefig(fname, bbox_inches='tight')


# function whose gradient we are approximating
def fun(x):
   return x**2

# Generate data

nsamples = 10000 # number of samples
x0       = 1     # point where gradient is to be estimated
sigma    = 1     # observation noise variance

for delta in [0.1,0.32,1.0]:
   x = x0 + delta * np.random.normal(size=nsamples)
   y = fun(x) + sigma * np.random.normal(size=nsamples)
   plot_grad_dens(x,y,"grad_%s.pdf" % delta)

