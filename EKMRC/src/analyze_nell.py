#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   analyze_nell.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/05/13 20:40:53
'''

# here put the import lib
import sys
sys.path.append(".")
import csv

nell_csv = "EKMRC/retrieve_concepts/retrieve_nell/NELL.08m.1115.esv.csv"
sample_nell_csv = "EKMRC/retrieve_concepts/retrieve_nell/NELL.sample.csv"

fin = open(nell_csv)
headers = []
rows = []

is_header = True
count = 0
for line in fin:
    items = line.strip().split('\t')
    if is_header:
        headers = items
        is_header = False
    else:
        rows.append(tuple(items))
    count += 1
    if count > 1000:
        break

with open(sample_nell_csv, 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)


    
