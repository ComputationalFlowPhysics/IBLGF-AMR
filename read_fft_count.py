#!/usr/bin/env python
import sys
import os
import re
import numpy as np

dir_name = './build'
filename = "parallel.out"

f= open(dir_name + '/' +  filename, 'r')
lines = f.readlines()

fft_stat = np.zeros(24)

for line in lines:
    match = re.match( r'I am a client on rank:(.*), with fft count =(.*)',line)
    if (match):
        rank = int(match.group(1))
        fft_count = int(match.group(2))
        fft_stat[rank] = fft_count

fft_stat.astype(int)
for a in fft_stat:
    print(a)



