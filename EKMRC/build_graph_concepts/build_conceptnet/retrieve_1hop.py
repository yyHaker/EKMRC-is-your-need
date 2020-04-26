#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   retrieve_1hop.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/04/07 16:33:58
'''

"""
检索知识图谱：对于某个token，分别检索出三部分：
1. sub-graph
    (1) 检索出头或者尾部包含该词的三元组，构建子图G
2. sub-graph triples
3. core_entity
"""

import sys
sys.path.append(".")

import random
import pickle
import argparse
import os
import nltk
import logging
import string
from tqdm import tqdm
from nltk.corpus import wordnet as wn

from multiprocessing import Pool

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

PROCESSES = 60

def extract_en_triples(conceptnet_path):
    """检索出所有英文的三元组"""
    en_triples = []
    with open(conceptnet_path, 'r', encoding="utf-8") as f:
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

                triple = (head, rel, tail)
                en_triples.append(triple)
    return en_triples


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


# def build_mapping(triples, entity_path, relation_path):
#     """build mapping of entities and triples"""
#     entity2id = {}
#     relation2id = {}
#     for triple in triples:
#         head, rel, tail = triple[0], triple[1], triple[2]
#         if head not in entity2id.keys():
#             entity2id[head] = len(entity2id)
#         if tail not in entity2id.keys():
#             entity2id[tail] = len(entity2id)
#         if rel not in relation2id.keys():
#             relation2id[rel] = len(relation2id)
#     with open(entity_path, 'w') as f_e:
#         for entity, idx in entity2id.items():
#             f_e.write(entity + " " + str(idx))
#             f_e.write('\n')
#     with open(relation_path, 'w') as f_r:
#         for relation, idx in relation2id.items():
#             f_r.write(relation + " " + str(idx))
#             f_r.write('\n')
#     id2entity = {v:k for k,v in entity2id.items()}
#     id2relation = {v:k for k,v in relation2id.items()}
#     return entity2id, id2entity, relation2id, id2relation


def get_concept_mapping(entity_path, relation_path):
    """read entity and relation mapping file"""
    entity2id = {}
    relation2id = {}
    with open(entity_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            ls = line.split(" ")
            # pass first line
            if len(ls) <= 1:
                continue
            entity = ls[0].strip()
            idx = int(ls[1].strip())
            entity2id[entity] = idx
    
    with open(relation_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            ls = line.split(" ")
            # pass first line
            if len(ls) <= 1:
                continue
            rel = ls[0].strip()
            idx = int(ls[1].strip())
            relation2id[rel] = idx
    return entity2id, relation2id
    

def search_triples(token, conceptnet_triples):
    """检索出头或者尾部包含该词的三元组"""
    triples = []
    core_entitys = set()
    # search triples
    for triple in conceptnet_triples:
        head, rel, tail = triple[0], triple[1], triple[2]
        if token in head.split("_") or token in tail.split("_"):
            triples.append(triple)
            if token in head.split("_"):
                core_entitys.add(head)
            if token in tail.split("_"):
                core_entitys.add(tail)
    # define core entity, choose the shortest
    core_entitys = list(core_entitys)
    if len(core_entitys) != 0:
        min_len = len(core_entitys[0])
        min_entity = core_entitys[0]
        for entity in core_entitys:
            if len(entity) < min_len:
                min_len = len(entity)
                min_entity = entity
        core_entity = min_entity
    else:
        core_entity = None
    return triples, core_entity


def search_triple_neighbor(cur_triple, conceptnet_triples):
    """检索出三元组的相邻的三元组"""
    neighbor_triples = []
    cur_head, cur_rel, cur_tail = cur_triple[0], cur_triple[1], cur_triple[2]
    for triple in conceptnet_triples:
        if triple == cur_triple:
            continue
        head, rel, tail = triple[0], triple[1], triple[2]
        if cur_head == head or cur_head == tail or cur_tail == head or cur_tail == tail:
            neighbor_triples.append(triple)
    return neighbor_triples


def build_graph(triples):
    """连接相同的实体构建子图, 返回子图G"""
    # x : [num_nodes, num_node_features]
    # edge : [2, num_edges]
    # edge_attr : [num_edges, num_edge_features]
    nodes = []
    edges = []
    edges_attr = []
    token_triples = []
    for triple in triples:
        head, rel, tail = triple[0], triple[1], triple[2]
        # remove empty entity triple
        if head == "" or head == " ":
            continue
        if tail == "" or tail == " ":
            continue
        # add nodes
        if head not in nodes:
            nodes.append(head)
        if tail not in nodes:
            nodes.append(tail)
        # add edge
        edges.append([head, tail])
        edges.append([tail, head])
        edges_attr.append(rel)
        token_triples.append(triple)
    return nodes, edges, edges_attr, token_triples


def build_graph_for_token(token, conceptnet_triples):
    """根据给定的token，构建子图"""
    contained_triples, core_entity = search_triples(token, conceptnet_triples)
    nodes, edges, edges_attr, token_triples = build_graph(contained_triples)
    return nodes, edges, edges_attr, token_triples, core_entity


def retrieve_tokens_graph(index, token_part, conceptnet_triples, stopwords, args):
    """retrieve tokens graph"""
    logger.info("begin run function {} at process {}".format(retrieve_tokens_graph, os.getpid()))
    token2datas = {}
    for token in tqdm(token_part):
        if token in set(string.punctuation):
            logger.info('{} is punctuation, skipped!'.format(token))
            # punctuation_cnt += 1
            continue        
        if args.no_stopwords and token in stopwords:
            logger.info('{} is stopword, skipped!'.format(token))
            # stopword_cnt += 1
            continue
        if args.ignore_length > 0 and len(token) <= args.ignore_length:
            logger.info('{} is too short, skipped!'.format(token))
            continue
        # build graph for token here
        nodes, edges, edges_attr, token_triples, core_entity = build_graph_for_token(token, conceptnet_triples)
        token2data = {}
        token2data["sub_graph"] = (nodes, edges, edges_attr)
        token2data["graph_triples"] = token_triples
        token2data["core_entity"] = core_entity
        token2datas[token] = token2data
        
    with open(os.path.join(args.output_dir, 'retrived_token_graphs_{}.data'.format(index)), 'wb') as fout:
        pickle.dump(token2datas, fout)    
    logger.info('Finished dumping retrieved token graphs {}'.format(index))
    

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def retrieved_entity_rel_emb(token2datas, entity2id, relation2id, entity_emb, relation_emb):
    """retrieve entity and relation embeddings"""
    entity2emb = {}
    relation2emb = {}
    for token, data in token2datas.items():
        graph_triples = data["graph_triples"]
        for triple in graph_triples:
            head, rel, tail = triple[0], triple[1], triple[2]
            if head not in entity2emb:
                entity2emb[head] = entity_emb[entity2id[head]]
            if rel not in relation2emb:
                relation2emb[rel] = relation_emb[relation2id[rel]]
            if tail not in entity2emb:
                entity2emb[tail] = entity_emb[entity2id[tail]]
    return entity2emb, relation2emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_token', type=str, default='EKMRC/data/ReCoRD_tokenization/tokens_self/train.tokenization.cased.data', 
                                                                                help='token file of train set')
    parser.add_argument('--eval_token', type=str, default='EKMRC/data/ReCoRD_tokenization/tokens_self/dev.tokenization.cased.data',
                                                                                help='token file of dev set')

    parser.add_argument('--conceptnet_path', type=str, default='EKMRC/data/conceptnet/conceptNet_process.txt', help='conceptnet triple path')
    parser.add_argument('--entity_path', type=str, default='EKMRC/build_graph_concepts/concept_embs/entity2id.txt', help="entity2id path")
    parser.add_argument('--relation_path', type=str, default='EKMRC/build_graph_concepts/concept_embs/relation2id.txt', help="relation2id path")

    parser.add_argument('--entity_emb_path', type=str, default='EKMRC/build_graph_concepts/concept_embs/entity_emb.pkl', help="entity emb path")
    parser.add_argument('--relation_emb_path', type=str, default='EKMRC/build_graph_concepts/concept_embs/rel_emb.pkl', help="relation emb path")

    parser.add_argument('--entity2emb_path', type=str, default='EKMRC/build_graph_concepts/concept_embs/entity2emb.txt', help="entity2emb path")
    parser.add_argument('--relation2emb_path', type=str, default='EKMRC/build_graph_concepts/concept_embs/relation2emb.txt', help='relation2emb path')

    parser.add_argument('--output_dir', type=str, default='EKMRC/build_graph_concepts/retrieve_result/one_hop', help='output directory')
    
    parser.add_argument('--no_stopwords', action='store_true', default=True, help='ignore stopwords')
    parser.add_argument('--ignore_length', type=int, default=0, help='ignore words with length <= ignore_length')
    args = parser.parse_args()

    # load ConceptNet here
    logger.info("Begin loading concept triples...")
    conceptnet_triples = extract_triples(args.conceptnet_path)
    logger.info('Finished loading concept english triples.')

    logger.info("sample five triples...")
    for i in range(5):
        triple = random.choice(conceptnet_triples)
        logger.info(triple)

    # # build mappings of entities and relations(all ConceptNet)
    # entity2id, id2entity, relation2id, id2relation = build_mapping(conceptnet_triples, args.entity_path, args.relation_path)
    # logger.info("Finished mapping of relations and entities.")

    # get concept mapping
    logger.info("get concept mapping...")
    entity2id, relation2id = get_concept_mapping(args.entity_path, args.relation_path)
    
    # load pickled samples
    logger.info('Begin to load tokenization results...')
    train_samples = pickle.load(open(args.train_token, 'rb'))
    dev_samples = pickle.load(open(args.eval_token, 'rb'))
    logger.info('Finished loading tokenization results.')
    
    # build token set
    all_token_set = set()
    for sample in train_samples + dev_samples:
        for token in sample['query_tokens'] + sample['document_tokens']:
            all_token_set.add(token)
    logger.info('Finished making tokenization results into token set.')

    # load stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    logger.info('Finished loading stopwords list.')

    # mk directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # retrive neighbor triples and build sub-graph
    logger.info('Begin to retrieve neighbor triples and build sub-graph...')
    # token2graph = dict()
    # stopword_cnt = 0
    # punctuation_cnt = 0
    all_token_set = list(all_token_set)

    # split all_token_set to processes parts and deal with multi-processing
    all_token_parts = []
    part_token_nums = int(len(all_token_set) / PROCESSES)
    for i in range(PROCESSES):
        if i != PROCESSES - 1:
            cur_token_set = all_token_set[i * part_token_nums: (i+1) * part_token_nums]
        else:
            cur_token_set = all_token_set[i * part_token_nums: ]
        all_token_parts.append(cur_token_set)
    
    # multi-processing
    logger.info("Begin to deal with {} processes...".format(PROCESSES))
    p = Pool(PROCESSES)
    for i, part in enumerate(all_token_parts):
        p.apply_async(retrieve_tokens_graph, args=(i, part, conceptnet_triples, stopwords, args,))
    p.close()
    p.join()
    logger.info("all processes done!")
    
    # combine all results
    logger.info('Finished retrieving token graphs, combine all result...')
    token2datas = {}
    for i in range(PROCESSES):
        with open(os.path.join(args.output_dir, 'retrived_token_graphs_{}.data'.format(i)), 'rb') as fin:
            token2data = pickle.load(fin)
            token2datas.update(token2data)
    logger.info("combine all results done!")
    logger.info('{} / {} tokens retrieved at lease 1 graph.'.format(len(token2datas), len(all_token_set)))

    with open(os.path.join(args.output_dir, 'retrived_token_graphs_1hop.data'), 'wb') as fout:
        pickle.dump(token2datas, fout)    
    logger.info('Finished dumping retrieved token graphs.')

    # with open(os.path.join(args.output_dir, 'retrived_token_graphs_1hop.data'), 'rb') as f_in:
    #     token2datas = pickle.load(f_in)
    
    logger.info("save retrieved entity and relation embeddings...")
    with open(args.entity_emb_path, 'rb') as f1:
        entity_emb = pickle.load(f1)
    with open(args.relation_emb_path, 'rb') as f2:
        relation_emb = pickle.load(f2)

    entity2emb, relation2emb = retrieved_entity_rel_emb(token2datas, entity2id, relation2id, entity_emb, relation_emb)

    with open(args.entity2emb_path, 'w', encoding='utf-8') as f:
        for entity, emb in entity2emb.items():
            assert len(emb) == 100
            if entity == "" or entity == " ":
                logger.info("empty entity: {}".format(entity))
            f.write(entity + " " + " ".join(map(str, emb)) + "\n")
    with open(args.relation2emb_path, 'w', encoding="utf-8") as f:
        for rel, emb in relation2emb.items():
            assert len(emb) == 100
            f.write(rel + " " + " ".join(map(str, emb)) + "\n")
    logger.info("For all KG, {}/{} retrieved entities used, {}/{} retrieved relations used.".format(
        len(entity2emb), len(entity_emb), len(relation2emb), len(relation_emb)))


if __name__ == '__main__':
    main()