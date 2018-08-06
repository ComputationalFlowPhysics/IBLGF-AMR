# Use Python3!!!!!!!!!!!!!!!!!!!! python 2 doesn't give right integral....

import scipy.special as special
import scipy.integrate as integrate
import numpy as np
import math

from lgf_integrator import *

N=101
fname = open("lgf_table_100.hpp", "w")

for n1 in range(0,N):
    for n2 in range(0, n1+1):
        for n3 in range(0,n2+1):
            print(n1, n2, n3)

            ft = lambda t : f(n1, n2, n3, t)
            result= lgf_int(n1,n2,n3)

            fname.write("{:.20e},\n".format(result))


fname.close()
# Use Python3!!!!!!!!!!!!!!!!!!!! python 2 doesn't give right integral....
