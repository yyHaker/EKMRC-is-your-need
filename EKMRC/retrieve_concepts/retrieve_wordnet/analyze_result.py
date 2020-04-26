#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   analyze_result.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/04/07 15:54:59
'''

# here put the import lib
import pickle

with open("EKMRC/retrieve_concepts/retrieve_wordnet/output_record/retrived_synsets.data", "rb") as f_in:
    token2synsets = pickle.load(f_in)
    for token, synsets in token2synsets.items():
        print("For token: {}, synsets info: num_synsets: {}".format(token, len(synsets)))
