import scipy.special as special
import scipy.integrate as integrate
import numpy as np
import math


f = lambda n1, n2, n3, t : special.ive(n1,2*t) * special.ive(n2,2*t) *special.ive(n3,2*t)

def lgf_int(n1,n2,n3):
    ft = lambda t : f(n1, n2, n3, t)
    result,error = integrate.quad(ft, 0, np.inf, epsabs=1e-16,
                    epsrel=1e-15)

    if math.isnan(result): # ocasionally it has problem integrating. 
        dn = 0.0
        while math.isnan(result): # ocasionally it has problem integrating. 
            dn = dn + 1e-2
            print(dn)
            ft1 = lambda t : f(n1, n2, n3+dn, t)
            ft2 = lambda t : f(n1, n2, n3-dn, t)
            result1,error = integrate.quad(ft1, 0, np.inf)
            result2,error = integrate.quad(ft2, 0, np.inf)
            result = (result1 + result2)/2.0

    return result


