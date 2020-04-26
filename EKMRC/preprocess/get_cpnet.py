#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_cpnet.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2019/12/11 10:32:39
'''

# here put the import lib
import requests
import re

# obj = requests.get("http://api.conceptnet.io/c/en/book").json()
# print(obj.keys())
# print(obj["edges"][0])


def search_text_for_entity(entity, limit=20):
    """search surface text for an entity"""
    texts = []
    obj = requests.get(f"http://api.conceptnet.io/c/en/{entity}").json()
    pattern = re.compile(r'[\[\]\.]')
    for edge in obj["edges"]:
        text = edge["surfaceText"]
        # remove '[' ']' '.'
        text_ = re.sub(pattern, "", text) + "."
        texts.append(text_)
    return " ".join(texts)

texts = search_text_for_entity("ab_extra")
print(texts)

# p = re.compile(r'[\[\]\.]')
# res = re.sub(p, "@", "[sdc][xvb.")
# print(res)


