import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def likelihood(x,y,C,D):
    return np.exp(-(y-C*x)**2/(2*D**2))/np.sqrt(2*np.pi*D**2)


# parameters for likelihood
C = 1
D = 0.5

# prior gauss
mean = 0
cov = 1

# measurement
y = 4.0

"""
Setup prior
"""
x = np.linspace(-10,10,1000)
prior = norm.pdf(x,mean,cov)

"""
Bayesian update
"""

# Compute Kalman Gain
K = cov*C/(C*cov*C+D*D)

# update mean
mean = mean + K*(y - C*mean)

#update covariance
cov = cov - K*C*cov

posterior = norm.pdf(x,mean,cov)

"""
plot stuff
"""
like = likelihood(x,y,C,D)

plt.figure()
plt.plot(x,prior,label="prior")
plt.plot(x,like,label="like")
plt.plot(x,posterior,label="posterior")
plt.legend()
plt.show()





