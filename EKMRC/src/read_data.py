#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   read_data.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2019/12/10 10:17:35
'''
import sys
sys.path.append(".")
import pickle
import json

# data_path = "data/retrived_kb_data/wordnet_record/dev.retrieved_nell_concepts.data"
# wordnet_path = "data/retrived_kb_data/wordnet_record/retrived_synsets.data"

# with open(wordnet_path, "rb") as f, open("data/retrived_kb_data/wordnet_record/sample.data.json", "w") as f_out:
#     datas = pickle.load(f)
#     sample = datas[0]
#     json.dump(sample, f_out)

# import torch
# import torch.nn as nn
# m = nn.Bilinear(20, 30, 40)
# input1 = torch.rand(86, 20)
# input2 = torch.rand(128, 30)
# output = m(input1, input2)
# print(output.size())

retrieved_nell_concept_filepath = "EKMRC/kgs/NELL/retrieve_nell_res/bert-large-uncased-wwm/dev.retrieved_nell_concepts.data"
nell_retrieve_info = {}
for item in pickle.load(open(retrieved_nell_concept_filepath, 'rb')):
    nell_retrieve_info[item['id']] = item
    document_entities = item["document_entities"]
    for doc_entity in document_entities:
        entity_string = doc_entity["entity_string"]
        retrieved_concepts = doc_entity["retrieved_concepts"]
        if len(retrieved_concepts) != 0:
            print("entity string: {}, retrieved_concepts: {}".format(entity_string, "||".join(retrieved_concepts)))

# print(a)
# retrieved_wordnet_concept_filepath = "EKMRC/data/retrived_kb_data/wordnet_record/retrived_synsets.data"
# with open(retrieved_wordnet_concept_filepath, 'rb') as f:
#     datas = pickle.load(f)
#     print("hello")

# data_path = "EKMRC/data/ReCoRD_tokenization/tokens/dev.tokenization.cased.data"
# with open(data_path, 'rb') as f:
#     datas = pickle.load(f)
#     print("hello")
