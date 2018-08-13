import scipy.special as special
import scipy.integrate as integrate
import numpy as np
import math
import random

#///////
from lgf_asym import *
from lgf_integrator import *


N_min    = 50
N_max    = 400
n_sample = 5000

n_abs   = [.0] * n_sample
n_error_23 = [.0] * n_sample

for i in range(0, n_sample):
    n1 = random.randint(N_min, N_max)
    n2 = random.randint(N_min, N_max)
    n3 = random.randint(N_min, N_max)

    print(n1, n2, n3)

    a2 = asym_2(n1,n2,n3)
    a3 = asym_3(n1,n2,n3)

    n_abs[i] = (n1**2 + n2**2 + n3**2)**(1/2)
    n_error_23[i] = abs(a3-a2)
    print(abs(n_error_23[i])) # definition has a sign difference


import matplotlib.pyplot as plt
plt.plot(n_abs, n_error_23, 'ro')
plt.yscale('log')
plt.ylim(1e-18,1e-5)
plt.xlim(N_min, N_max * 1.7)
plt.grid(True)
plt.title('error of 2-term-asymptotic vs |n|')
plt.show()


