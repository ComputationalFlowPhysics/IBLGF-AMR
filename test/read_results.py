import sys
import os
import re

err = []

if len(sys.argv)>1:
    dir_name = sys.argv[1]
else:
    dir_name = 'sin_results'

for filename in os.listdir(dir_name):
    name = [int(s) for s in re.findall(r'\d+',filename)]
    b_extent = name[0]
    domain = name[0] *  name[1]

    f = open(dir_name + '/' +  filename, 'r')
    text = f.read()
    e_inf_s = re.findall(r'LInf = -?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', text)

    if len(e_inf_s) >0:
        print('-----------')
        print(filename)
        print(b_extent, domain)
        e_inf_s = e_inf_s[0]
        e_inf_s = e_inf_s[7:]
        e_inf = float(e_inf_s)
        print(e_inf)

        err.append( (b_extent, domain, e_inf))

    f.close()


err.sort(key=lambda tup: (tup[0], tup[1]))
for e in err:
    print(e)
for e in err:
    print(e[2])


