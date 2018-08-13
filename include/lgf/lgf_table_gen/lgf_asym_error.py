import scipy.special as special
import scipy.integrate as integrate
import numpy as np
import math
import random

#///////
from lgf_asym import *
from lgf_integrator import *


N_max    = 110
n_sample = 500

n_abs   = [.0] * n_sample
n_error_2 = [.0] * n_sample
n_error_3 = [.0] * n_sample

for i in range(0, n_sample):
    n1 = random.randint(50, N_max)
    n2 = random.randint(50, N_max)
    n3 = random.randint(50, N_max)

    print(n1, n2, n3)
    a_int = lgf_int(n1,n2,n3)

    a2 = asym_2(n1,n2,n3)
    a3 = asym_3(n1,n2,n3)
    print(a_int, a2, a3)

    n_abs[i] = (n1**2 + n2**2 + n3**2)**(1/2)
    n_error_2[i] = a_int + a2
    n_error_3[i] = a_int + a3
    print(abs(n_error_2[i])) # definition has a sign difference
    print(abs(n_error_3[i])) # definition has a sign difference


import matplotlib.pyplot as plt
plt.plot(n_abs, n_error_3, 'ro')
plt.yscale('log')
plt.ylim(1e-18,1e-10)
plt.grid(True)
plt.title('error of 3-term-asymptotic vs |n|')
plt.show()

