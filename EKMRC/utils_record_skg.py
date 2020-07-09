#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
@File    :   utils_record_skg.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/04/14 15:22:02
'''

# here put the import lib
""" Load SQuAD and ReCoRD dataset, add some relvant knowledge."""

import json
import logging
import math
import collections
from io import open
from tqdm import tqdm
import pickle
import os

import torch

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
from torch.utils.data import Dataset

from torch_geometric.data import Data

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 core_entity_ids,
                 graph_data_dict,
                #  graph_info_ids,
                #  nodes_ids,
                #  edges_ids,
                #  edges_attr_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.core_entity_ids = core_entity_ids
        self.graph_data_dict = graph_data_dict
        # self.example_graph_info = example_graph_info
        # self.graph_info_ids = graph_info_ids
        # self.nodes_ids=nodes_ids,
        # self.edges_ids=edges_ids,
        # self.edges_attr_ids=edges_attr_ids,
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class InputFeatures_both(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 wn_concept_ids,
                 nell_concept_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.wn_concept_ids = wn_concept_ids
        self.nell_concept_ids = nell_concept_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SelfDataset(Dataset):
    r"""Dataset wrapping tensors and python list.

    Each sample will be retrieved by indexing along the first dimension.
    All data should be tensors, except the last one is python list.

    Arguments:
        *datas (data): datas that have the same size of the first dimension.
    """

    def __init__(self, *datas):
        assert all(datas[0].size(0) == data.size(0) for data in datas[:-1])
        assert datas[0].size(0) == len(datas[-1])
        self.datas = datas

    def __getitem__(self, index):
        return tuple(data[index] for data in self.datas)

    def __len__(self):
        return self.datas[0].size(0)


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    # if (len(qa["answers"]) != 1) and (not is_impossible):
                    #     raise ValueError(
                    #         "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def read_record_examples(input_file, is_training, version_2_with_negative=False):
    """Read a ReCoRD json file into a list of ReCoRDExample."""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        # white space tokenization
        paragraph_text = entry["passage"]["text"].replace('\xa0', ' ')
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        for qa in entry["qas"]:
            qas_id = qa["id"]
            question_text = qa["query"].replace('\xa0', ' ')
            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False
            if is_training:
                if version_2_with_negative:
                    is_impossible = qa["is_impossible"]
                # if (len(qa["answers"]) != 1) and (not is_impossible):
                #     raise ValueError(
                #         "For training, each question should have exactly 1 answer."
                #     )
                if not is_impossible:
                    # just chose the first one?
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset +
                                                        answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(doc_tokens[start_position:(
                        end_position + 1)])
                    cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.info("Could not find answer: '%s' vs. '%s'",
                                actual_text, cleaned_answer_text)
                        continue
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            example = SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible)
            examples.append(example)

    return examples

'''for search related concepts here'''
def convert_examples_to_features(index, 
                                 examples, 
                                 tokenizer, 
                                 max_seq_length,
                                 doc_stride, 
                                 max_query_length, 
                                 is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', 
                                 sep_token='[SEP]', 
                                 pad_token=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=0, 
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False,
                                 tokenization_path=None,
                                 entity2id=None, 
                                 relation2id=None,
                                 entity_emb_mat=None,
                                 rel_emb_mat=None,
                                 retrieved_conceptnet_path=None,
                                 retrieved_wordnet_path=None,
                                 use_wordnet=False,
                                 use_conceptnet=True,
                                 example_cache_dir=None):
    """Loads a data file into a list of `InputBatch`s."""
    # parse some params
    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    '''load ReCoRD tokenization info'''
    all_tokenization_info = {}
    if is_training:
        tokenization_path = os.path.join(tokenization_path, "train.tokenization.cased.data")
    else:
        tokenization_path = os.path.join(tokenization_path, "dev.tokenization.cased.data")
    assert tokenization_path is not None
    for item in pickle.load(open(tokenization_path, 'rb')):
        all_tokenization_info[item['id']] = item
    
    '''load retrived knowledge(only one KB concept used)'''
    '''
        token: graph(nodes, edges, edges_attr)
    '''
    assert (use_wordnet and use_conceptnet) is False

    retrieve_graph_info = {}
    # max_concept_length = 0
    # max_nodes_num = 0
    # max_edges_num = 0
    # max_edges_attr_num = 0
    if use_wordnet:
        assert os.path.exists(retrieved_wordnet_path)
        # here is not relavant to "q_id"
        # retrieve_concept_info = pickle.load(open(retrieved_wordnet_concept_path, 'rb'))
        # max_concept_length = max([len(synsets) for synsets in retrieve_concept_info.values()])
    elif use_conceptnet:
        assert os.path.exists(retrieved_conceptnet_path)
        token2datas = pickle.load(open(retrieved_conceptnet_path, 'rb'))
        for token, data in token2datas.items():
            sub_graph_info = {}
            sub_graph_info["sub_graph"] = data["sub_graph"]
            sub_graph_info["core_entity"] = data["core_entity"]
            retrieve_graph_info[token] = sub_graph_info
            # get max num
            # max_nodes_num = max(len(sub_graph[0]), max_nodes_num)
            # max_edges_num = max(len(sub_graph[1]), max_edges_num)
            # max_edges_attr_num = max(len(sub_graph[2]), max_edges_attr_num)
        # for item in pickle.load(open(retrieved_nell_concept_path, 'rb')):
        #     retrieve_concept_info[item['id']] = item
        # max_concept_length = max([max([len(entity_info['retrieved_concepts']) for entity_info in item['query_entities'] + item['document_entities']]) 
        #                                 for qid, item in retrieve_concept_info.items() if len(item['query_entities'] + item['document_entities']) > 0])
    
    # here use (<pad_concept>: 0) padding
    assert entity2id is not None
    assert relation2id is not None

    def _lookup_conceptnet_graph_ids(sub_tokens, sub_to_ori_index, words, retrieve_graph_info):
        '''return list of graph info ids given input subword list'''
        graph_info_ids = []
        for index in range(len(sub_tokens)):
            original_word = words[sub_to_ori_index[index]]
            # if word are in upper case, we must lower it for retrieving
            retrieve_token = original_word.lower()
            sub_graph = {}
            nodes_ids, edges_ids, edges_attr_ids = [], [], []
            core_entity_id = 0
            if retrieve_token in retrieve_graph_info:
                sub_graph_info = retrieve_graph_info[retrieve_token]
                nodes, edges, edges_attr = sub_graph_info["sub_graph"]
                core_entity = sub_graph_info["core_entity"]
                if core_entity is not None:
                    nodes_ids = [entity2id[node] for node in nodes]
                    edges_ids = [[entity2id[item[0]], entity2id[item[1]]] for item in edges]
                    edges_attr_ids = [relation2id[edge_attr] for edge_attr in edges_attr]
                    core_entity_id = entity2id[core_entity]
                else:
                    core_entity_id = 0 # use 0 represent no sub-graph
            # graph info
            sub_graph["nodes_ids"] = nodes_ids
            sub_graph["edges_ids"] = edges_ids
            sub_graph["edges_attr_ids"] = edges_attr_ids
            sub_graph["core_entity_id"] = core_entity_id
            graph_info_ids.append(sub_graph)
        return graph_info_ids
    
    def _update_graph_dict(graph_info_dict, sub_graph):
        '''update graph_info_dict according to sub_graph'''
        core_entity = sub_graph["core_entity_id"]
        if core_entity not in graph_info_dict:
            graph_info_dict[core_entity] = sub_graph
        return graph_info_dict

    # def _lookup_nell_concept_ids(sub_tokens, sub_to_ori_index, words, nell_info):
    #     original_concept_ids = [[] for _ in range(len(words))]
    #     for entity_info in nell_info:
    #         for pos in range(entity_info['token_start'], entity_info['token_end'] + 1):
    #             original_concept_ids[pos] += [concept2id[category_name] for category_name in entity_info['retrieved_concepts']]
    #     # del the dulplicate
    #     for pos in range(len(original_concept_ids)):
    #         original_concept_ids[pos] = list(set(original_concept_ids[pos]))
    #     # copy origin concept_ids to sub
    #     concept_ids = [original_concept_ids[sub_to_ori_index[index]] for index in range(len(sub_tokens))]
    #     return concept_ids

    features = []
    for (example_index, example) in enumerate(tqdm(examples)):

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        # query_tokens = tokenizer.tokenize(example.question_text)

        # for query, construct origin(bert tokenization) and token(after word_piece_tokenization) mappings
        query_tokens = tokenizer.tokenize(example.question_text)

        # query_words = tokenizer.basic_tokenizer.tokenize(example.question_text)
        # tok_to_orig_index = [] # word_piece2origin inflection
        # orig_to_tok_index = [] # origin2word_piece inflection
        # query_tokens = [] # word_piece tokens
        # for (i, word) in enumerate(query_words):
        #     orig_to_tok_index.append(len(query_tokens))
        #     sub_tokens = tokenizer.tokenize(word)
        #     for sub_token in sub_tokens:
        #         tok_to_orig_index.append(i)
        #         query_tokens.append(sub_token)
        
        tokenization_info = all_tokenization_info[example.qas_id]

        # query_tokens: wordpiece tokenization
        # tokenization_info["query_tokens"]: white_space tokenization

        # not same here
        assert query_tokens == tokenization_info["query_subtokens"]
        # assert tok_to_orig_index == tokenization_info["query_sub_to_ori_index"]
        # assert query_words == tokenization_info["query_tokens"]

        query_tokens = tokenization_info["query_subtokens"]

        if use_wordnet:
            pass
            # query_concepts = _lookup_wordnet_concept_ids(query_tokens, tokenization_info["query_sub_to_ori_index"], tokenization_info["query_tokens"],
            #     retrieve_concept_info)
        elif use_conceptnet:
            query_graphs = _lookup_conceptnet_graph_ids(query_tokens, tokenization_info["query_sub_to_ori_index"], 
                                    tokenization_info["query_tokens"], retrieve_graph_info)

        # control the question length after word_piece_tokenization
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
            query_graphs = query_graphs[0:max_query_length]

        # for doc, construct origin(bert tokenization) and token(after word_piece_tokenization) mappings
        tok_to_orig_index = [] # word_piece2origin inflection
        orig_to_tok_index = [] # origin2word_piece inflection
        all_doc_tokens = [] # word_piece tokens
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # not samme here
        # assert all_doc_tokens == tokenization_info['document_subtokens']

        all_doc_tokens = tokenization_info["document_subtokens"]
        
        if use_wordnet:
            pass
            # doc_concepts = _lookup_wordnet_concept_ids(all_doc_tokens, tokenization_info["document_sub_to_ori_index"],
            #  tokenization_info["document_tokens"], retrieve_concept_info)
        elif use_conceptnet:
            doc_graphs = _lookup_conceptnet_graph_ids(all_doc_tokens, tokenization_info["document_sub_to_ori_index"],
             tokenization_info["document_tokens"], retrieve_graph_info)
        
        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        
        # # get cls and sep token id
        # seg_ids = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
        # cls_token_id = seg_ids[0]
        # sep_token_id = seg_ids[1]

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            # set there parts here to save sub-graph info
            graph_info_dict = {} # (key: core_entity_id, value: sub_graph)，combine all token sub-graph to a big graph at last
            core_entity_ids = [] # every token has a core entity
            # nodes_ids = []       # sub-graph nodes      padding use 0
            # edges_ids = []       # sub-graph edges      padding use 0
            # edges_attr_ids = []  # sub-graph edges ids  padding use 0

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                # nodes_ids.append([])
                # edges_ids.append([])
                # edges_attr_ids.append([])
                # graph_info_ids.append([[], [], []])
                core_entity_ids.append(0)
                p_mask.append(0)
                cls_index = 0

            # XLNet: P SEP Q SEP CLS
            # Others: CLS Q SEP P SEP
            if not sequence_a_is_doc:
                # Query
                tokens += query_tokens
                segment_ids += [sequence_a_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)
                # for graph in query_graphs:
                #     nodes_ids.append(graph[0])
                #     edges_ids.append(graph[1])
                #     edges_attr_ids.append(graph[2])
                # graph_info_ids += query_graphs
                for sub_graph in query_graphs:
                    core_entity_ids.append(sub_graph["core_entity_id"])
                    graph_info_dict = _update_graph_dict(graph_info_dict, sub_graph)

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
                # nodes_ids.append([])
                # edges_ids.append([])
                # edges_attr_ids.append([])
                # graph_info_ids.append([[], [], []])
                core_entity_ids.append(0)


            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                sub_graph = doc_graphs[split_token_index]
                # nodes_ids.append(sub_graph[0])
                # edges_ids.append(sub_graph[1])
                # edges_attr_ids.append(sub_graph[2])
                # graph_info_ids.append(doc_graphs[split_token_index])
                core_entity_ids.append(sub_graph["core_entity_id"])
                graph_info_dict = _update_graph_dict(graph_info_dict, sub_graph)
                p_mask.append(0)
            paragraph_len = doc_span.length

            if sequence_a_is_doc:
                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
                # nodes_ids.append([])
                # edges_ids.append([])
                # edges_attr_ids.append([])
                # graph_info_ids.append([[], [], []])
                core_entity_ids.append(0)

                tokens += query_tokens
                segment_ids += [sequence_b_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)
                # for graph in query_graphs:
                #     nodes_ids.append(graph[0])
                #     edges_ids.append(graph[1])
                #     edges_attr_ids.append(graph[2])
                for sub_graph in query_graphs:
                    core_entity_ids.append(sub_graph["core_entity_id"])
                    graph_info_dict = _update_graph_dict(graph_info_dict, sub_graph)
                # graph_info_ids += query_graphs

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)
            # nodes_ids.append([])
            # edges_ids.append([])
            # edges_attr_ids.append([])
            # graph_info_ids.append([[], [], []])
            core_entity_ids.append(0)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token
                # nodes_ids.append([])
                # edges_ids.append([])
                # edges_attr_ids.append([])
                # graph_info_ids.append([[], [], []])
                core_entity_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)
                # padding core_entity_ids
                core_entity_ids.append(0)
            
            # combine all token graph to a big graph
            example_graph_info = {}
            example_graph_info["nodes_ids"] = []
            example_graph_info["edges_ids"] = []
            example_graph_info["edges_attr_ids"] = []
            # add an empty graph
            example_graph_info["nodes_ids"] += [0]
            example_graph_info["edges_ids"] += [[0, 0]]
            example_graph_info["edges_attr_ids"] += [0]
            for core_entity, sub_graph in graph_info_dict.items():
                example_graph_info["nodes_ids"] += sub_graph["nodes_ids"]
                example_graph_info["edges_ids"] += sub_graph["edges_ids"]
                example_graph_info["edges_attr_ids"] += sub_graph["edges_attr_ids"]
            
            nodes_ids = example_graph_info["nodes_ids"]
            edges_ids = example_graph_info["edges_ids"]
            edges_attr_ids = example_graph_info["edges_attr_ids"]

            # get entity and rel embedding
            nodes_emb = torch.tensor([entity_emb_mat[node_id] for node_id in nodes_ids], dtype=torch.float)
            edges_attr_emb = torch.tensor([rel_emb_mat[edge_attr_id] for edge_attr_id in edges_attr_ids], dtype=torch.float)

            # mapping node to new id
            map_node_to_new_id = {}
            for node_id in nodes_ids:
                if node_id not in map_node_to_new_id:
                    map_node_to_new_id[node_id] = len(map_node_to_new_id)
            
            edges_index = torch.tensor([[map_node_to_new_id[item[0]], map_node_to_new_id[item[1]]] for item in edges_ids], dtype=torch.long).t()

            core_entity_ids = [map_node_to_new_id[core_id] for core_id in core_entity_ids]

            # graph_data = Data(x=nodes_emb, edge_index=edges_index, edge_attr=edges_attr_emb)
            graph_data_dict = {"nodes_features": nodes_emb, "edges_index": edges_index, "edges_attr": edges_attr_emb}

            # for concept to pad
            # here zero represents no concept
            # for cindex in range(len(concept_ids)):
            #     concept_ids[cindex] = concept_ids[cindex] + [0] * (max_concept_length - len(concept_ids[cindex]))
            #     concept_ids[cindex] = concept_ids[cindex][: max_concept_length]
            # if len(concept_ids) < max_seq_length:
            #     for _ in range(max_seq_length - len(concept_ids)):
            #         concept_ids.append([0] * max_concept_length)

            # for graph info ids to pad
            # if len(graph_info_ids) < max_seq_length:
            #     for _ in range(max_seq_length - len(graph_info_ids)):
            #         graph_info_ids.append([[], [], []])
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(core_entity_ids) == max_seq_length
            # assert len(graph_info_ids) == max_seq_length
            # assert len(nodes_ids) == max_seq_length
            # assert len(edges_ids) == max_seq_length
            # assert len(edges_attr_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    if sequence_a_is_doc:
                        doc_offset = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            # if example_index < 2:
            #     logger.info("*** Example ***")
            #     logger.info("unique_id: %s" % (unique_id))
            #     logger.info("example_index: %s" % (example_index))
            #     logger.info("doc_span_index: %s" % (doc_span_index))
            #     logger.info("tokens: %s" % " ".join(tokens))
            #     logger.info("token_to_orig_map: %s" % " ".join([
            #         "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            #     logger.info("token_is_max_context: %s" % " ".join([
            #         "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
            #     ]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info(
            #         "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     logger.info("graph info: ")
            #     logger.info("core entity ids: %s" % " ".join([str(x) for x in core_entity_ids]))
            #     # logger.info("   subgraph node ids: %s" % " ".join(["{}:{}".format(tidx, list(filter(lambda index: index != 0, x[0]))) for tidx, x in enumerate(graph_info_ids)]))
            #     # logger.info("   subgraph edge ids: %s" % " ".join(["{}:{}".format(tidx, list(filter(lambda index: index != 0, x))) for tidx, x in enumerate(edges_attr_ids)]))
            #     # logger.info("   node ids: %s" % " ".join(["{}:{}".format(tidx, list(filter(lambda index:index != 0, x[0]))) for tidx, x in enumerate(graph_info_ids)]))
            #     # logger.info("   edges_attr ids: %s" % " ".join(["{}:{}".format(tidx, list(filter(lambda index:index != 0, x[2]))) for tidx, x in enumerate(graph_info_ids)]))
            #     if is_training and span_is_impossible:
            #         logger.info("impossible example")
            #     if is_training and not span_is_impossible:
            #         answer_text = " ".join(tokens[start_position:(end_position + 1)])
            #         logger.info("start_position: %d" % (start_position))
            #         logger.info("end_position: %d" % (end_position))
            #         logger.info(
            #             "answer: %s" % (answer_text))
            
            # add concept_ids following
            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,  # 0
                    input_mask=input_mask, # 1
                    segment_ids=segment_ids, # 2
                    core_entity_ids=core_entity_ids,
                    graph_data_dict=graph_data_dict,
                    # example_graph_info=example_graph_info,
                    # graph_info_ids=graph_info_ids,
                    # nodes_ids=nodes_ids,
                    # edges_ids=edges_ids,
                    # edges_attr_ids=edges_attr_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position, # 3
                    end_position=end_position, # 4
                    is_impossible=span_is_impossible))
            unique_id += 1
    
    # save features to file
    # with open(os.path.join(example_cache_dir, 'examples_part_{}.data'.format(index)), 'wb') as fout:
    #     pickle.dump(features, fout)
    # logger.info("finished dump features to examples_part_{}.data".format(index))
    return features


'''for search related concepts(two concepts) here'''
def convert_examples_to_features_both(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False,
                                 tokenization_path=None,
                                 wn_concept2id=None,
                                 nell_concept2id=None,
                                 retrieved_nell_concept_path=None,
                                 retrieved_wordnet_concept_path=None,
                                 use_wordnet=True,
                                 use_nell=True):
    """Loads a data file into a list of `InputBatch`s."""
    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    '''load ReCoRD tokenization info'''
    all_tokenization_info = {}
    if is_training:
        tokenization_path = os.path.join(tokenization_path, "train.tokenization.cased.data")
    else:
        tokenization_path = os.path.join(tokenization_path, "dev.tokenization.cased.data")
    assert tokenization_path is not None
    for item in pickle.load(open(tokenization_path, 'rb')):
        all_tokenization_info[item['id']] = item
    
    '''load retrived knowledge(both wordnet and nell used here)'''
    assert (use_wordnet or use_nell) is True

    if use_wordnet:
        assert retrieved_wordnet_concept_path is not None
        retrieved_wordnet_concept_path = os.path.join(retrieved_wordnet_concept_path, "retrived_synsets.data")
    
    if use_nell:
        assert retrieved_nell_concept_path is not None
        if is_training:
            retrieved_nell_concept_path = os.path.join(retrieved_nell_concept_path, "train.retrieved_nell_concepts.data")
        else:
            retrieved_nell_concept_path = os.path.join(retrieved_nell_concept_path, "dev.retrieved_nell_concepts.data")

    wn_retrieve_concept_info = {}
    nell_retrieve_concept_info = {}
    wn_max_concept_length = 0
    nell_max_concept_length = 0

    if use_wordnet:
        assert os.path.exists(retrieved_wordnet_concept_path)
        # here is not relavant to "q_id"
        wn_retrieve_concept_info = pickle.load(open(retrieved_wordnet_concept_path, 'rb'))
        wn_max_concept_length = max([len(synsets) for synsets in wn_retrieve_concept_info.values()])

    if use_nell:
        assert os.path.exists(retrieved_nell_concept_path)
        for item in pickle.load(open(retrieved_nell_concept_path, 'rb')):
            nell_retrieve_concept_info[item['id']] = item
        nell_max_concept_length = max([max([len(entity_info['retrieved_concepts']) for entity_info in item['query_entities'] + item['document_entities']]) 
                                        for qid, item in nell_retrieve_concept_info.items() if len(item['query_entities'] + item['document_entities']) > 0])
    
    assert wn_concept2id is not None
    assert nell_concept2id is not None

    '''return list of concept ids given input subword list'''
    def _lookup_nell_concept_ids(sub_tokens, sub_to_ori_index, words, nell_info):
        original_concept_ids = [[] for _ in range(len(words))]
        for entity_info in nell_info:
            for pos in range(entity_info['token_start'], entity_info['token_end'] + 1):
                original_concept_ids[pos] += [nell_concept2id[category_name] for category_name in entity_info['retrieved_concepts']]
        # del the dulplicate
        for pos in range(len(original_concept_ids)):
            original_concept_ids[pos] = list(set(original_concept_ids[pos]))
        # copy origin concept_ids to sub
        concept_ids = [original_concept_ids[sub_to_ori_index[index]] for index in range(len(sub_tokens))]
        return concept_ids
    
    def _lookup_wordnet_concept_ids(sub_tokens, sub_to_ori_index, words, wordnet_info):
        # for wordnet, just find the synset name
        concept_ids = []
        for index in range(len(sub_tokens)):
            original_word = words[sub_to_ori_index[index]]
            # if word are in upper case, we must lower it for retrieving
            retrieve_token = original_word.lower()
            if retrieve_token in wordnet_info:
                concept_ids.append([wn_concept2id[synset_name] for synset_name in wordnet_info[retrieve_token]])
            else:
                concept_ids.append([])
        return concept_ids

    features = []
    for (example_index, example) in enumerate(tqdm(examples)):

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        # query_tokens = tokenizer.tokenize(example.question_text)

        # for query, construct origin(bert tokenization) and token(after word_piece_tokenization) mappings
        query_tokens = tokenizer.tokenize(example.question_text)

        # query_words = tokenizer.basic_tokenizer.tokenize(example.question_text)
        # tok_to_orig_index = [] # word_piece2origin inflection
        # orig_to_tok_index = [] # origin2word_piece inflection
        # query_tokens = [] # word_piece tokens
        # for (i, word) in enumerate(query_words):
        #     orig_to_tok_index.append(len(query_tokens))
        #     sub_tokens = tokenizer.tokenize(word)
        #     for sub_token in sub_tokens:
        #         tok_to_orig_index.append(i)
        #         query_tokens.append(sub_token)
        
        tokenization_info = all_tokenization_info[example.qas_id]

        # query_tokens: wordpiece tokenization
        # tokenization_info["query_tokens"]: white_space tokenization

        # not same here
        assert query_tokens == tokenization_info["query_subtokens"]
        # assert tok_to_orig_index == tokenization_info["query_sub_to_ori_index"]
        # assert query_words == tokenization_info["query_tokens"]

        query_tokens = tokenization_info["query_subtokens"]

        if use_wordnet:
            wn_query_concepts = _lookup_wordnet_concept_ids(query_tokens, tokenization_info["query_sub_to_ori_index"], tokenization_info["query_tokens"],
             wn_retrieve_concept_info)

        if use_nell:
            nell_query_concepts = _lookup_nell_concept_ids(query_tokens, tokenization_info["query_sub_to_ori_index"], tokenization_info["query_tokens"], 
            nell_retrieve_concept_info[example.qas_id]["query_entities"])
        
        # control the question length after word_piece_tokenization
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0: max_query_length]
            wn_query_concepts = wn_query_concepts[0: max_query_length]
            nell_query_concepts = nell_query_concepts[0: max_query_length]

        # for doc, construct origin(bert tokenization) and token(after word_piece_tokenization) mappings
        tok_to_orig_index = [] # word_piece2origin inflection
        orig_to_tok_index = [] # origin2word_piece inflection
        all_doc_tokens = [] # word_piece tokens
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # not samme here
        # assert all_doc_tokens == tokenization_info['document_subtokens']

        all_doc_tokens = tokenization_info["document_subtokens"]
        
        if use_wordnet:
            wn_doc_concepts = _lookup_wordnet_concept_ids(all_doc_tokens, tokenization_info["document_sub_to_ori_index"],
             tokenization_info["document_tokens"], wn_retrieve_concept_info)

        if use_nell:
            nell_doc_concepts = _lookup_nell_concept_ids(all_doc_tokens, tokenization_info["document_sub_to_ori_index"],
             tokenization_info["document_tokens"], nell_retrieve_concept_info[example.qas_id]["document_entities"])
        
        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            wn_concept_ids = []
            nell_concept_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                wn_concept_ids.append([])
                nell_concept_ids.append([])
                p_mask.append(0)
                cls_index = 0

            # XLNet: P SEP Q SEP CLS
            # Others: CLS Q SEP P SEP
            if not sequence_a_is_doc:
                # Query
                tokens += query_tokens
                segment_ids += [sequence_a_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)
                wn_concept_ids += wn_query_concepts
                nell_concept_ids += nell_query_concepts

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
                wn_concept_ids.append([])
                nell_concept_ids.append([])

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                wn_concept_ids.append(wn_doc_concepts[split_token_index])
                nell_concept_ids.append(nell_doc_concepts[split_token_index])
                p_mask.append(0)
            paragraph_len = doc_span.length

            if sequence_a_is_doc:
                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
                wn_concept_ids.append([])
                nell_concept_ids.append([])

                tokens += query_tokens
                segment_ids += [sequence_b_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)
                wn_concept_ids += wn_query_concepts
                nell_concept_ids += nell_query_concepts

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)
            wn_concept_ids.append([])
            nell_concept_ids.append([])

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token
                wn_concept_ids.append([])
                nell_concept_ids.append([])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            # for concept to pad
            # here zero represents no concept
            for concept_ids, max_concept_length in zip((wn_concept_ids, nell_concept_ids), (wn_max_concept_length, nell_max_concept_length)):
                for cindex in range(len(concept_ids)):
                    concept_ids[cindex] = concept_ids[cindex] + [0] * (max_concept_length - len(concept_ids[cindex]))
                    concept_ids[cindex] = concept_ids[cindex][: max_concept_length]

                if len(concept_ids) < max_seq_length:
                    for _ in range(max_seq_length - len(concept_ids)):
                        concept_ids.append([0] * max_concept_length)
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(wn_concept_ids) == max_seq_length
            assert len(nell_concept_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    if sequence_a_is_doc:
                        doc_offset = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 2:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("wn_concept_ids: %s" % " ".join(["{}:{}".format(tidx, list(filter(lambda index:index != 0, x))) for tidx, x in enumerate(wn_concept_ids)]))
                logger.info("nell_concept_ids: %s" % " ".join(["{}:{}".format(tidx, list(filter(lambda index:index != 0, x))) for tidx, x in enumerate(nell_concept_ids)]))
                
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))
            
            # add concept_ids following
            features.append(
                InputFeatures_both(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,  # 0
                    input_mask=input_mask, # 1
                    segment_ids=segment_ids, # 2
                    wn_concept_ids=wn_concept_ids,
                    nell_concept_ids=nell_concept_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position, # 3
                    end_position=end_position, # 4
                    is_impossible=span_is_impossible))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


# For XLNet (and XLM which uses the same head)
RawResultExtended = collections.namedtuple("RawResultExtended",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


def write_predictions_extended(all_examples, all_features, all_results, n_best_size,
                                max_answer_length, output_prediction_file,
                                output_nbest_file,
                                output_null_log_odds_file, orig_data_file,
                                start_n_top, end_n_top, version_2_with_negative,
                                tokenizer, verbose_logging):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.

        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index",
        "start_log_prob", "end_log_prob"])

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

    logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            # 
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer.do_lower_case,
                                        verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_log_prob=-1e6,
                end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    with open(orig_data_file, "r", encoding='utf-8') as reader:
        orig_data = json.load(reader)["data"]

    qid_to_has_ans = make_qid_to_has_ans(orig_data)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(orig_data, all_predictions)
    out_eval = {}

    find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw, scores_diff_json, qid_to_has_ans)

    return out_eval


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
