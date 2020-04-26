#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   retrieve.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/03/16 19:14:19
'''
"""
构图方法：对于某个token
1. 检索出头或者尾部包含该词的三元组
2. 检索出三元组的相邻的三元组
3. 链接相同的实体，构建子图G
"""

import sys
sys.path.append(".")

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
        for entity, idx in entity2id.items():
            f_e.write(entity + " " + str(idx))
            f_e.write('\n')
    with open(relation_path, 'w') as f_r:
        for relation, idx in relation2id.items():
            f_r.write(relation + " " + str(idx))
            f_r.write('\n')
    id2entity = {v:k for k,v in entity2id.items()}
    id2relation = {v:k for k,v in relation2id.items()}
    return entity2id, id2entity, relation2id, id2relation
    

def search_triples(token, conceptnet_triples):
    """检索出头或者尾部包含该词的三元组"""
    triples = []
    for triple in conceptnet_triples:
        head, rel, tail = triple[0], triple[1], triple[2]
        if token in head.split("_") or token in tail.split("_"):
            triples.append(triple)
    return triples


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


def build_graph(triples, entity2id, relation2id):
    """连接相同的实体构建子图, 返回子图G"""
    # x : [num_nodes, num_node_features]
    # edge_index : [2, num_edges]
    # edge_attr : [num_edges, num_edge_features]
    nodes = []
    edge_index = []
    edges = []
    token_triples = []
    for triple in triples:
        head, rel, tail = triple[0], triple[1], triple[2]
        head_id = entity2id[head]
        rel_id = relation2id[rel]
        tail_id = entity2id[tail]
        # add nodes
        if head_id not in nodes:
            nodes.append(head_id)
        if tail_id not in nodes:
            nodes.append(tail_id)
        # add edge
        edge_index.append([head_id, tail_id])
        edge_index.append([tail_id, head_id])
        edges.append(rel_id)
    return nodes, edge_index, edges, token_triples


def build_graph_for_token(token, conceptnet_triples, entity2id, relation2id):
    """根据给定的token，构建子图"""
    triples_dict = {}
    contained_triples = search_triples(token, conceptnet_triples)
    for triple in contained_triples:
        neighbor_triples = search_triple_neighbor(triple, conceptnet_triples)
        for neighbor_triple in neighbor_triples:
            if neighbor_triple not in triples_dict:
                triples_dict[neighbor_triple] = neighbor_triple
    triples = list(triples_dict.keys())
    nodes, edge_index, edges, token_triples = build_graph(triples, entity2id, relation2id)
    return nodes, edge_index, edges, token_triples


def retrieve_tokens_graph(index, token_part, conceptnet_triples, entity2id, relation2id, stopwords, args):
    """retrieve tokens graph"""
    logger.info("begin run function {} at process {}".format(retrieve_tokens_graph, os.getpid()))
    token2graph = {}
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
        nodes, edge_index, edges, token_triples = build_graph_for_token(token, conceptnet_triples, entity2id, relation2id)
        token2data = {}
        token2data["token2graph"] = (nodes, edge_index, edges)
        token2data["token2triples"] = token_triples
        token2graph[token] = token2data
    
    with open(os.path.join(args.output_dir, 'retrived_token_graphs_{}.data'.format(index)), 'wb') as fout:
        pickle.dump(token2graph, fout)    
    logger.info('Finished dumping retrieved token graphs {}.'.format(index))
    

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_token', type=str, default='EKMRC/data/ReCoRD_tokenization/tokens_self/train.tokenization.cased.data', 
                                                                                help='token file of train set')
    parser.add_argument('--eval_token', type=str, default='EKMRC/data/ReCoRD_tokenization/tokens_self/dev.tokenization.cased.data',
                                                                                help='token file of dev set')
    parser.add_argument('--conceptnet_path', type=str, default='EKMRC/data/conceptnet/conceptnet-assertions-5.6.0.csv',
                                                                                help='conceptnet triple path')
    parser.add_argument('--entity_path', type=str, default='EKMRC/build_graph_concepts/entity2id.txt', help="entity2id path")
    parser.add_argument('--relation_path', type=str, default='EKMRC/build_graph_concepts/relation2id.txt', help="relation2id path")
    parser.add_argument('--output_dir', type=str, default='EKMRC/build_graph_concepts/retrieve_result', help='output directory')
    parser.add_argument('--no_stopwords', action='store_true', default=True, help='ignore stopwords')
    parser.add_argument('--ignore_length', type=int, default=0, help='ignore words with length <= ignore_length')
    args = parser.parse_args()

    # load ConceptNet here
    conceptnet_triples = extract_en_triples(args.conceptnet_path)
    logger.info('Finished loading concept english triples.')

    # build mappings of entities and relations
    entity2id, id2entity, relation2id, id2relation = build_mapping(conceptnet_triples, args.entity_path, args.relation_path)
    logger.info("Finished mapping of relations and entities.")
    
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
    token2graph = dict()
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
        p.apply_async(retrieve_tokens_graph, args=(i, part, conceptnet_triples, entity2id, relation2id, stopwords, args,))
    p.close()
    p.join()
    logger.info("all processes done!")
    
    # combine all results
    logger.info('Finished retrieving token graphs, combine all result...')
    token2graphs = {}
    for i in range(PROCESSES):
        with open(os.path.join(args.output_dir, 'retrived_token_graphs_{}.data'.format(i)), 'rb') as fin:
            token2graph = pickle.load(fin)
            token2graphs.update(token2graph)
    logger.info("combine all results done!")
    logger.info('{} / {} tokens retrieved at lease 1 graph.'.format(len(token2graphs), len(all_token_set)))

    with open(os.path.join(args.output_dir, 'retrived_token_graphs.data'), 'wb') as fout:
        pickle.dump(token2graphs, fout)    
    logger.info('Finished dumping retrieved token graphs.')

if __name__ == '__main__':
    main()