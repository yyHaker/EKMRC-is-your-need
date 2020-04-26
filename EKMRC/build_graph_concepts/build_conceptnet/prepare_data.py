#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/04/10 16:43:23
'''

"""处理数据成OpenKE可以训练的格式"""

# here put the import lib
import os
import logging
import argparse


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def get_concept_mapping(entity_path, relation_path):
    """read entity and relation mapping file"""
    entity2id = {}
    relation2id = {}
    with open(entity_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            ls = line.split(" ")
            entity = ls[0].strip()
            idx = int(ls[1].strip())
            entity2id[entity] = idx
    
    with open(relation_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            ls = line.split(" ")
            rel = ls[0].strip()
            idx = int(ls[1].strip())
            relation2id[rel] = idx
    return entity2id, relation2id


def extract_triples(conceptnet_path):
    """检索出conceptnet中的三元组"""
    conceptnet_triples = []
    with open(conceptnet_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            ls = line.split(",")
            head = ls[0].strip()
            rel = ls[1].strip()
            tail = ls[2].strip()

            triple = (head, rel, tail)
            conceptnet_triples.append(triple)
    return conceptnet_triples

def build_mapping(triples, entity_path, relation_path):
    """build mapping of entities and triples"""
    entity2id = {}
    relation2id = {}
    for triple in triples:
        head, rel, tail = triple[0], triple[1], triple[2]
        if head not in entity2id.keys():
            entity2id[head] = len(entity2id)
        if tail not in entity2id.keys():
            entity2id[tail] = len(entity2id)
        if rel not in relation2id.keys():
            relation2id[rel] = len(relation2id)

    with open(entity_path, 'w') as f_e:
        f_e.write(str(len(entity2id)) + "\n")
        for entity, idx in entity2id.items():
            f_e.write(entity + " " + str(idx))
            f_e.write('\n')

    with open(relation_path, 'w') as f_r:
        f_r.write(str(len(relation2id)) + "\n")
        for relation, idx in relation2id.items():
            f_r.write(relation + " " + str(idx))
            f_r.write('\n')
    id2entity = {v:k for k,v in entity2id.items()}
    id2relation = {v:k for k,v in relation2id.items()}
    return entity2id, id2entity, relation2id, id2relation

def get_train_file(conceptnet_triples, entity2id, relation2id, train2id_path):
    """得到训练文件, 格式为(e1, e2, rel)"""
    triple_num = len(conceptnet_triples)
    with open(train2id_path, 'w', encoding="utf-8") as f:
        f.write(str(triple_num))
        f.write('\n')
        for triple in conceptnet_triples:
            head_id = entity2id[triple[0]]
            rel_id = relation2id[triple[1]]
            tail_id = entity2id[triple[2]]
            f.write(str(head_id) + " " + str(tail_id) + " " + str(rel_id))
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conceptnet_path', type=str, default='EKMRC/data/conceptnet/conceptNet_process.txt', help='conceptnet triple path')

    parser.add_argument('--entity_path', type=str, default='EKMRC/build_graph_concepts/concept_embs/entity2id.txt', help="entity2id path")
    parser.add_argument('--relation_path', type=str, default='EKMRC/build_graph_concepts/concept_embs/relation2id.txt', help="relation2id path")
    parser.add_argument('--output_path', type=str, default='EKMRC/build_graph_concepts/concept_embs/train2id.txt', help="train2id path")
    args = parser.parse_args()

    logger.info("load concept triples...")
    conceptnet_triples = extract_triples(args.conceptnet_path)

    # build mappings of entities and relations(all ConceptNet)
    logger.info("build concept mapping...")
    entity2id, id2entity, relation2id, id2relation = build_mapping(conceptnet_triples, args.entity_path, args.relation_path)
    logger.info("Finished mapping of relations and entities.")

    logger.info("get train file format...")
    get_train_file(conceptnet_triples, entity2id, relation2id, args.output_path)
    logger.info("get train file done!")




