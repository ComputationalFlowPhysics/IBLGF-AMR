import scipy.special as special
import scipy.integrate as integrate
import numpy as np
import math

f = lambda n1, n2, n3,t : special.ive(n1,2*t) * special.ive(n2,2*t) *special.ive(n3,2*t)

def asym_2(n1,n2,n3):
    n_abs = math.sqrt(n1**2+n2**2+n3**2)
    tmp = -1/4/math.pi/n_abs
    tmp = tmp - (n1**4 + n2**4 + n3**4 - 3*n1**2*n2**2 - 3*n2**2*n3**2 -
            3*n3**2*n1**2)/16/math.pi/n_abs**7
    return tmp

def asym_3(n1,n2,n3):
    n_abs = math.sqrt(n1**2+n2**2+n3**2)
    s1 = asym_2(n1,n2,n3)
    s2 = 3*(   23 * (n1**8 + n2**8 + n3**8) 
            - 244 * (n2**6 * n3**2 + n3**6 * n2**2 + 
                     n1**6 * n2**2 + n2**6 * n1**2 + 
                     n1**6 * n3**2 + n3**6 * n1**2  
                     )
            + 621 * (n1**4 * n2**4 + n2**4 * n3**4 + n3**4 * n1**4 )
            - 228 * (n1**4 * n2**2 * n3**2 
                     + n1**2 * n2**4 * n3**2
                     + n1**2 * n2**2 * n3**4))

    coef = 1/(768 * math.pi) * 8
    s2 = -s2 / n_abs**13 /4 * coef
    
    return s1 + s2
    


NN = 1
for n1 in range(1,NN):
    for n2 in range(1,NN):
        for n3 in range(1,NN):
            print(n1, n2, n3)
            
            ft = lambda t : f(n1, n2, n3, t)
            result,error = integrate.quad(ft, 0, np.inf, epsabs=1e-15)

            a2 = asym_2(n1,n2,n3)
            a3 = asym_3(n1,n2,n3)
            print(result, a2, a3)
            print(result + a2) 
            print(result + a3) # definition has a sign difference



