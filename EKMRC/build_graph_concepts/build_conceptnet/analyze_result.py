#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   analyze_result.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/04/07 14:23:36
'''

# here put the import lib
import sys
sys.path.append(".")
import pickle

"""Analyze the result"""

with open("EKMRC/build_graph_concepts/retrieve_result/one_hop/retrived_token_graphs_1hop.data", "rb") as f_in:
    token2data = pickle.load(f_in)
    count = 0
    for token, data in token2data.items():
        sub_graph = data["sub_graph"]
        graph_triples = data["graph_triples"]
        nodes, edge_index, edges = sub_graph
        print("For token: {}, subgraph info: num_nodes: {}, num_edges: {}, graph triples: {}".format(token, len(nodes), len(edges), len(graph_triples)))
        # print("token2triples: ", token2triples)

        count += 1
        if count == 100:
            break


