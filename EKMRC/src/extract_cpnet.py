#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   extract_cpnet.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2019/12/04 13:43:14
'''

# here put the import lib
import json
import requests
import re
from random import shuffle
from collections import defaultdict

conceptnet_path = "../data/conceptnet/conceptnet-assertions-5.6.0.csv"
conceptnet_entities_file = "../data/conceptnet/cpnet-entities.txt"

conceptnet_en_raw_text_train_path = "../data/conceptnet/conceptnet-assertions-5.6.0.csv.en.train.raw.txt"
conceptnet_en_raw_text_dev_path = "../data/conceptnet/conceptnet-assertions-5.6.0.csv.en.dev.raw.txt"

conceptnet_en_entity_text_train_path = "../data/conceptnet/conceptnet-assertions-5.6.0.csv.en.train.entity.context.txt"
conceptnet_en_entity_text_dev_path = "../data/conceptnet/conceptnet-assertions-5.6.0.csv.en.dev.entity.context.txt"


# relation maps of relation to raw text
relation_mapping = {
    "atlocation": "at location",
    "capableof": "capable of",
    "locatednear": "located near",
    "causes": "causes",
    "causesdesire": "causes desire",
    "motivatedbygoal": "motivated by goal",
    "createdby": "created by",
    "desires": "desires",
    "antonym": "antonym",
    "distinctfrom": "distinct from",
    "derivedfrom": "derived from",
    "etymologicallyderivedfrom": "etymologically derived from",
    "hascontext": "has context",
    "hasproperty": "has property",
    "hassubevent": "has subevent",
    "hasfirstsubevent": "has first subevent",
    "haslastsubevent": "has last subevent",
    "hasprerequisite": "has prerequisite",
    "entails": "entails",
    "mannerof": "manner of",
    "isa": "is a", 
    "instanceof": "instance of",
    "definedas": "defined as",
    "madeof": "made of",
    "notcapableof": "not capable of",
    "notdesires": "not desires",
    "partof": "part of",
    "hasa": "has a",
    "relatedto": "related to",
    "etymologicallyrelatedto": "etymologically related to",
    "similarto": "similar to",
    "synonym": "synonym",
    "usedfor": "used for",
    "receivesaction": "receives action",
    "formof": "form of",
    "nothasproperty": "not has property",
    "symbolof": "symbol of",
    "capital": "capital",
    "field": "field",
    "genre": "genre"
}

def extract_english_raw_texts():
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the raw text format for each line: <head> + <relation> + <tail>, and transfer to a natual language.
    """
    # conceptnet triples raw text
    cpnet_en_raw_text = []

    # conceptnet entity context
    cpnet_en_entity_context = []

    with open(conceptnet_path, encoding="utf8") as f:
        for line in f.readlines():
            ls = line.split('\t')
            if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = ls[1].split("/")[-1].lower()
                head = del_pos(ls[2]).split("/")[-1].lower()
                tail = del_pos(ls[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue

                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue

                # transfer to raw text
                head_text = head.replace("_", " ")
                tail_text = tail.replace("_", " ")

                if rel not in relation_mapping:
                    continue

                rel_text = relation_mapping[rel]
                raw_text = head_text + " " + rel_text + " " + tail_text
                cpnet_en_raw_text.append(raw_text)
    
    # split to train and test
    shuffle(cpnet_en_raw_text)
    train_size = int(len(cpnet_en_raw_text) * 0.9)
    cpnet_en_raw_train_text = cpnet_en_raw_text[:train_size]
    cpnet_en_raw_dev_text = cpnet_en_raw_text[train_size:]

    with open(conceptnet_en_raw_text_train_path, "w", encoding="utf8") as f:
        f.write("\n".join(cpnet_en_raw_train_text))
    
    with open(conceptnet_en_raw_text_dev_path, "w", encoding="utf8") as f:
        f.write("\n".join(cpnet_en_raw_dev_text))


# def extract_english_raw_entities():
#     """extract entities from conceptnet"""
#     entities = set()
#     with open(conceptnet_path, encoding="utf-8") as f:
#         for line in f.readlines():
#             ls = line.split('\t')
#             if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en'):
#                 """
#                 Some preprocessing:
#                     - Remove part-of-speech encoding.
#                     - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
#                     - Lowercase for uniformity.
#                 """
#                 rel = ls[1].split("/")[-1].lower()
#                 head = del_pos(ls[2]).split("/")[-1].lower()
#                 tail = del_pos(ls[3]).split("/")[-1].lower()

#                 if not head.replace("_", "").replace("-", "").isalpha():
#                     continue

#                 if not tail.replace("_", "").replace("-", "").isalpha():
#                     continue
                
#                 entities.add(head)
#                 entities.add(tail)
    
#     with open(conceptnet_entities_file, 'w', encoding='utf-8') as f_out:
#         f_out.write('\n'.join(list(entities)))
#     print("write {} entities to file done!".format(len(list(entities))))


def extract_english_raw_texts_for_entity(limit=20):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the raw text format for each line: <head> + <relation> + <tail>, and transfer to a natual language.
    根据每一个实体去寻找三元组，再转换为自然语言.
    """
    raw_texts = []
    entity_contexts_dict = defaultdict(list)
    with open(conceptnet_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ls = line.split('\t')
            if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = ls[1].split("/")[-1].lower()
                head = del_pos(ls[2]).split("/")[-1].lower()
                tail = del_pos(ls[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue

                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue

                # transfer to raw text
                head_text = head.replace("_", " ")
                tail_text = tail.replace("_", " ")

                if rel not in relation_mapping:
                    continue

                rel_text = relation_mapping[rel]
                raw_text = head_text + " " + rel_text + " " + tail_text + "."

                # add to dict
                entity_contexts_dict[head].append(raw_text)
                entity_contexts_dict[tail].append(raw_text)

    for entity, contexts in entity_contexts_dict.items():
        raw_texts.append((entity + '\n' + " ".join(contexts)))
    
    # split to train and test, and write to file
    shuffle(raw_texts)
    train_size = int(len(raw_texts) * 0.9)
    cpnet_en_entity_train_text = raw_texts[:train_size]
    cpnet_en_entity_dev_text = raw_texts[train_size:]

    with open(conceptnet_en_entity_text_train_path, "w", encoding="utf8") as f:
        f.write("\n\n".join(cpnet_en_entity_train_text))
    
    with open(conceptnet_en_entity_text_dev_path, "w", encoding="utf8") as f:
        f.write("\n\n".join(cpnet_en_entity_dev_text))
    
    print("get total {} contexts".format(len(raw_texts)))
    

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s

# def search_text_for_entity(entity, limit=20):
#     """search surface text for an entity"""
#     texts = []
#     obj = requests.get(f"http://api.conceptnet.io/c/en/{entity}").json()
#     pattern = re.compile(r'[\[\]\.]')
#     for edge in obj["edges"]:
#         text = edge["surfaceText"]
#         # remove '[' ']' '.'
#         text_ = re.sub(pattern, "", text) + "."
#         texts.append(text_)
#     return " ".join(texts)


if __name__ == "__main__":
    # extract_english_raw_texts()
    # extract_english_raw_entities()
    extract_english_raw_texts_for_entity()