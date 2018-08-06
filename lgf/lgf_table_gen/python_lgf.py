import scipy.special as special
import scipy.integrate as integrate
import numpy as np

f = lambda n1, n2, n3,t : special.ive(n1,2*t) * special.ive(n2,2*t) *special.ive(n3,2*t)

N=101
fname = open("lgf_table_100.hpp", "w")

for n1 in range(0,N):
    for n2 in range(0, n1+1):
        for n3 in range(0,n2+1):
            ft = lambda t : f(n1, n2, n3, t)
            result,error = integrate.quad(ft, 0, np.inf, epsabs=1e-16,
                    epsrel=1e-15)
            #print(n1, n2, n3)
            #print(result, error)

            fname.write("{:.20e},\n".format(result))

fname.close()

