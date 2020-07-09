'''tokenization ReCoRD for easy to search related concepts.'''
from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open
from tqdm import tqdm
import pickle
import os
import argparse

import sys
sys.path.append(".")
sys.path.append("EKMRC")

logger = logging.getLogger(__name__)


from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer,
                                  DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'ktnet': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
}

class ReCoRDExample(object):
    """
    A single training/test example for the ReCoRD dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 passage_entities,
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
        self.passage_entities = passage_entities

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
        
        # load entities in passage
        passage_entities = []
        for entity in entry['passage']['entities']:
            entity_start_offset = entity['start']
            entity_end_offset = entity['end']
            # some error labeled entities in record dataset
            if entity_end_offset < entity_start_offset:
                continue
            entity_text = paragraph_text[entity_start_offset: entity_end_offset + 1]
            passage_entities.append({
                'orig_text': entity_text,
                'start_position': char_to_word_offset[entity_start_offset],
                'end_position': char_to_word_offset[entity_end_offset]
            })


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

            example = ReCoRDExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                passage_entities=passage_entities,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible)
            examples.append(example)

    return examples


def tokenization_on_examples(examples, tokenizer):
    """for further tokenization process when generating features, for easy to search related concepts"""
    tokenization_result = []
    for example in tqdm(examples):
        # do tokenization on raw question text
        query_subtokens = []
        query_sub_to_ori_index = [] # mapping from sub-token index to token index
        query_tokens = tokenizer.basic_tokenizer.tokenize(example.question_text)
        for index, token in enumerate(query_tokens):
            for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                query_subtokens.append(sub_token)
                query_sub_to_ori_index.append(index)
    
        # do tokenization on whitespace tokenized document
        document_tokens = []
        document_subtokens = []
        document_sub_to_ori_index = []
        document_up_to_ori_index = [] # map unpunc token index to tokenized token index
        for unpunc_tokenized_tokens in example.doc_tokens:
            tokens = tokenizer.basic_tokenizer.tokenize(unpunc_tokenized_tokens) # do punctuation tokenization
            document_up_to_ori_index.append(len(document_tokens))
            for token in tokens:
                for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                    document_subtokens.append(sub_token)
                    document_sub_to_ori_index.append(len(document_tokens))
                document_tokens.append(token)
        
        # generate token-level document entity index
        document_entities = []
        for entity in example.passage_entities:
            entity_start_position = document_up_to_ori_index[entity['start_position']]
            entity_end_position = None
            if entity['end_position'] < len(example.doc_tokens) - 1:
                entity_end_position = document_up_to_ori_index[entity['end_position'] + 1] - 1
            else:
                entity_end_position = len(document_tokens) - 1
            (entity_start_position, entity_end_position) = _improve_entity_span(
                document_tokens, entity_start_position, entity_end_position, tokenizer, entity['orig_text'])
            document_entities.append((entity['orig_text'], entity_start_position, entity_end_position)) # ('Trump', 10, 10)
        
        # match query to passage entities
        query_entities = match_query_entities(query_tokens, document_entities, document_tokens) # [('trump', 10, 10)]
        
        tokenization_result.append({
            'id': example.qas_id,
            'query_tokens': query_tokens,
            'query_subtokens': query_subtokens,
            'query_sub_to_ori_index': query_sub_to_ori_index,
            'query_entities': query_entities,
            'document_tokens': document_tokens,
            'document_subtokens': document_subtokens,
            'document_entities': document_entities,
            'document_sub_to_ori_index': document_sub_to_ori_index,
        })
    return tokenization_result

def _improve_entity_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_entity_text):
    """Returns token-level tokenized entity spans that better match the annotated entity."""
    tok_entity_text = " ".join(tokenizer.basic_tokenizer.tokenize(orig_entity_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_entity_text:
                return (new_start, new_end)

    return (input_start, input_end)

def match_query_entities(query_tokens, document_entities, document_tokens):
    # transform query_tokens list into a whitespace separated string
    query_string = " ".join(query_tokens)
    offset_to_tid_map = []
    tid = 0
    for char in query_string:
        offset_to_tid_map.append(tid)
        if char == ' ':
            tid += 1

    # transform entity_tokens into whitespace separated strings
    entity_strings = set()
    for document_entity in document_entities:
        entity_tokens = document_tokens[document_entity[1]: document_entity[2] + 1]
        entity_strings.add(" ".join(entity_tokens))
    
    # do matching
    results = []
    for entity_string in entity_strings:
        start = 0
        while True:
            pos = query_string.find(entity_string, start)
            if pos == -1:
                break
            token_start, token_end = offset_to_tid_map[pos], offset_to_tid_map[pos] + entity_string.count(' ')
            # assure the match is not partial match (eg. "ville" matches to "danville")
            if " ".join(query_tokens[token_start: token_end + 1]) == entity_string:
                results.append((token_start, token_end))
            start = pos + len(entity_string)
    
    # filter out a result span if it's a subspan of another span
    no_subspan_results = []
    for result in results:
        if not any([_is_real_subspan(result[0], result[1], other_result[0], other_result[1]) for other_result in results]):
            no_subspan_results.append((" ".join(query_tokens[result[0]: result[1] + 1]), result[0], result[1]))
    assert len(no_subspan_results) == len(set(no_subspan_results))

    return no_subspan_results

def _is_real_subspan(start, end, other_start, other_end):
    return (start >= other_start and end < other_end) or (start > other_start and end <= other_end)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default="EKMRC/data/ReCoRD/train.json", type=str,
                        help="ReCoRD json for training.")
    parser.add_argument("--dev_file", default="EKMRC/data/ReCoRD/dev.json", type=str,
                        help="ReCoRD json for dev")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="bert-large-uncased-whole-word-masking", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default="EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm", type=str,
                        help="The output directory where the tokenization result saved.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    # make dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("tokenizationing train record examples...")
    train_file = args.train_file
    train_examples = read_record_examples(input_file=args.train_file, is_training=True, version_2_with_negative=False)
    train_tokenization_res = tokenization_on_examples(train_examples, tokenizer)
    if "uncased" in args.model_name_or_path:
        train_tokenization_path = os.path.join(args.output_dir, "train.tokenization.uncased.data")
    elif "cased" in args.model_name_or_path:
        train_tokenization_path = os.path.join(args.output_dir, "train.tokenization.cased.data")
    with open(train_tokenization_path, 'wb') as f:
        pickle.dump(train_tokenization_res, f)
    
    logger.info("tokenizationing dev record examples...")
    dev_file = args.dev_file
    dev_examples = read_record_examples(input_file=args.dev_file, is_training=True, version_2_with_negative=False)
    dev_tokenization_res = tokenization_on_examples(dev_examples, tokenizer)
    if "uncased" in args.model_name_or_path:
        dev_tokenization_path = os.path.join(args.output_dir, "dev.tokenization.uncased.data")
    elif "cased" in args.model_name_or_path:
        dev_tokenization_path = os.path.join(args.output_dir, "dev.tokenization.cased.data")
    with open(dev_tokenization_path, 'wb') as f:
        pickle.dump(dev_tokenization_res, f)

    logger.info("tokenization all data done!")

if __name__ == "__main__":
    main()
    
    

    
