import sys
import os
import re

err = []
err_2 = []
err_lap_2 = []
tot_time =[]

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
    e_2_s   = re.findall(r'L2   = -?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', text)
    e_inf_s = re.findall(r'LInf = -?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', text)
    e_lap_2_s = re.findall(r'L2_source   =-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', text)


    if len(e_inf_s) >0:
        print('-----------')
        print(filename)
        print(b_extent, domain)
        e_inf_s = e_inf_s[0]
        e_inf_s = e_inf_s[7:]
        e_inf = float(e_inf_s)
        err.append( (b_extent, domain, e_inf))

    if len(e_2_s) >0:
        print('L2 -----------')
        e_2_s = e_2_s[0]
        e_2_s = e_2_s[7:]
        e_2 = float(e_2_s)
        err_2.append( (b_extent, domain, e_2))

    if len(e_lap_2_s) >0:
        print('L2_source  -----------')
        e_lap_2_s = e_lap_2_s[0]
        print(e_lap_2_s)
        e_lap_2_s = e_lap_2_s[14:]
        e_lap_2 = float(e_lap_2_s)
        err_lap_2.append( (b_extent, domain, e_lap_2))

    time_s = re.findall(r'time = -?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', text)
    sum_t = 0.0
    for t_s in time_s:
        t = float(t_s[7:])
        sum_t += t

    tot_time.append( (b_extent, domain, sum_t))

    f.close()


err.sort(key=lambda tup: (tup[0], tup[1]))
err_2.sort(key=lambda tup: (tup[0], tup[1]))
tot_time.sort(key=lambda tup: (tup[0], tup[1]))
err_lap_2.sort(key=lambda tup: (tup[0], tup[1]))

for e in err:
    print(e)

print("Linf")
for e in err:
    print(e[2])

print("L2")
for e in err_2:
    print(e[2])

print("time")
for t in tot_time:
    print(t[2])

print("lap_source")
for e in err_lap_2:
    print(e[2])

