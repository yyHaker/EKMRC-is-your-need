#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   load_cache_record_skg_examples.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/04/27 09:15:42
'''

# here put the import lib
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
@File    :   run_record_skg.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/04/13 17:17:54
'''

# here put the import lib
"""使用ConceptNet处理ReCoRD数据集，得到examples"""

import sys
sys.path.append(".")

import argparse
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import glob
import timeit

import heapq
import pickle

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  SKGNetForQuestionAnswering,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer,
                                  DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from utils_record_skg import (read_record_examples, convert_examples_to_features, read_squad_examples,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended, SelfDataset)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
from utils_record_evaluate import EVAL_OPTS, main as evaluate_on_record

from multiprocessing import Pool
PROCESSES = 60

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'skgnet': (BertConfig, SKGNetForQuestionAnswering, BertTokenizer),
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
}


# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.n_gpu > 0:
#         torch.cuda.manual_seed_all(args.seed)

def read_concept_embedding(embedding_path):
    """read concept emb from file"""
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[0].split(' ')[1:])
    for line in info:
        length = len(line.strip().split(" "))
        assert length == 101
    n_concept = len(info)
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    for line in info:
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]] 
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    embedding_mat = np.array(embedding_mat, dtype=np.float32)
    return id2concept, concept2id, embedding_mat


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # load concept embedding mat(here should add both entity and rel emb)
    if args.use_conceptnet:
        logger.info("use conceptnet...")
        logger.info("load entity embeddings...")
        id2entity, entity2id, entity_emb_mat = read_concept_embedding("EKMRC/build_graph_concepts/concept_embs/entity2emb.txt")
        logger.info("load relation embeddings...")
        id2relation, relation2id, rel_emb_mat = read_concept_embedding("EKMRC/build_graph_concepts/concept_embs/relation2emb.txt")

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cpnet_name = "wordnet" if args.use_wordnet else "conceptnet"

    # change the cache name according to the args.model_name_or_path
    cache_name = list(filter(None, args.model_name_or_path.split('/'))).pop()
    if "stage1" in cache_name:
        if "ktnet" in cache_name and "large" in cache_name:
            cache_name = "bert-large-cased"
        elif "ktnet" in cache_name and "base" in cache_name:
            cache_name = "bert-base-cased"

    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        cache_name,
        cpnet_name,
        str(args.max_seq_length)))
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if output_examples:
            examples = read_record_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=args.version_2_with_negative)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_record_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=args.version_2_with_negative)
        
        # examples = examples[:10] # for debug here.
        # use multi-process to process data
        all_examples_parts = []
        part_example_nums = int(len(examples) / PROCESSES)
        for i in range(PROCESSES):
            if i != PROCESSES -1:
                cur_example_part = examples[i * part_example_nums: (i+1) * part_example_nums]
            else:
                cur_example_part = examples[(i+1) * part_example_nums: ]
            all_examples_parts.append(cur_example_part)
        
        # multi-processing
        logger.info("Begin to deal with {} processes...".format(PROCESSES))
        p = Pool(PROCESSES)
        for i, part in enumerate(all_examples_parts):
            p.apply_async(convert_examples_to_features, 
                                                args=(
                                                i,
                                                part,
                                                tokenizer,
                                                args.max_seq_length,
                                                args.doc_stride,
                                                args.max_query_length,
                                                not evaluate,),
                                                kwds={
                                                    "cls_token_segment_id": 2 if args.model_type in ['xlnet'] else 0,
                                                    "pad_token_segment_id": 3 if args.model_type in ['xlnet'] else 0,
                                                    "cls_token_at_end": True if args.model_type in ['xlnet'] else False,
                                                    "sequence_a_is_doc": True if args.model_type in ['xlnet'] else False,
                                                    "tokenization_path": args.tokenization_path,
                                                    "entity2id": entity2id,
                                                    "relation2id": relation2id,
                                                    "entity_emb_mat": entity_emb_mat,
                                                    "rel_emb_mat": rel_emb_mat,
                                                    "retrieved_conceptnet_path": args.retrieved_conceptnet_path,
                                                    "retrieved_wordnet_path": args.retrieved_wordnet_path,
                                                    "use_wordnet": args.use_wordnet,
                                                    "use_conceptnet": args.use_conceptnet,
                                                    "example_cache_dir": args.example_cache_dir
                                                })
        p.close()
        p.join()
        logger.info("all processes done!")

        all_examples = []
        for i in range(PROCESSES):
            with open(os.path.join(args.example_cache_dir, "examples_part_{}.data".format(i)), 'rb') as f:
                examples_part = pickle.load(f)
                all_examples += examples_part
        features = all_examples
        logger.info("combine all process examples, total examples is {}".format(features))
            
        # features = convert_examples_to_features(index=index,
        #                                         examples=examples,
        #                                         tokenizer=tokenizer,
        #                                         max_seq_length=args.max_seq_length,
        #                                         doc_stride=args.doc_stride,
        #                                         max_query_length=args.max_query_length,
        #                                         is_training=not evaluate,
        #                                         cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
        #                                         pad_token_segment_id=3 if args.model_type in ['xlnet'] else 0,
        #                                         cls_token_at_end=True if args.model_type in ['xlnet'] else False,
        #                                         sequence_a_is_doc=True if args.model_type in ['xlnet'] else False,
        #                                         tokenization_path=args.tokenization_path,
        #                                         entity2id=entity2id,
        #                                         relation2id=relation2id,
        #                                         entity_emb_mat=entity_emb_mat,
        #                                         rel_emb_mat=rel_emb_mat,
        #                                         retrieved_conceptnet_path=args.retrieved_conceptnet_path,
        #                                         retrieved_wordnet_path=args.retrieved_wordnet_path,
        #                                         use_wordnet=args.use_wordnet,
        #                                         use_conceptnet=args.use_conceptnet,
        #                                         example_cache_dir=args.example_cache_dir)

        
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    ## graph info here
    all_core_entity_ids = torch.tensor([f.core_entity_ids for f in features], dtype=torch.long)
    all_graph_data_dict = [f.graph_data_dict for f in features]
    # all_example_graph_info = [f.example_graph_info for f in features]
    # all_concept_ids = torch.tensor([f.concept_ids for f in features], dtype=torch.long)
    # all_graph_info_ids = [f.graph_info_ids for f in features]
    # all_nodes_ids = torch.tensor([f.nodes_ids for f in features], dtype=torch.long)
    # all_edges_ids = torch.tensor([f.edges_ids for f in features], dtype=torch.long)
    # all_edges_attr_ids = torch.tensor([f.edges_attr_ids for f in features], dtype=torch.long)

    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = SelfDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask, all_core_entity_ids, all_graph_data_dict)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = SelfDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask, all_core_entity_ids, all_graph_data_dict)

    if output_examples:
        return dataset, examples, features
    return dataset, entity_emb_mat, rel_emb_mat


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default="EKMRC/data/ReCoRD_graph/train.json", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default="EKMRC/data/ReCoRD_graph/dev.json", type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default="skgnet", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default="EKMRC/results/debug_ReCoRD", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # for concept params
    parser.add_argument("--embedding_path", default="EKMRC/data/retrived_kb_data/kb_embeddings/nell_concept2vec.txt", type=str,
                        help="the pretrained concept embedding path")
    parser.add_argument("--retrieved_conceptnet_path", default="EKMRC/build_graph_concepts/retrieve_result/one_hop/retrived_token_graphs_1hop.data", type=str,
                        help="retrievedd nell concept path")
    parser.add_argument("--retrieved_wordnet_path", default="EKMRC/data/retrived_kb_data/wordnet_record", type=str,
                        help="retrieved wordnet concept path")
    parser.add_argument("--use_wordnet", action='store_true', help="if use wordnet")
    parser.add_argument("--use_conceptnet", action='store_true', default=True, help="is use conceptnet")
    
    # data path
    parser.add_argument("--tokenization_path", default="EKMRC/data/ReCoRD_tokenization/tokens_self", type=str,
                        help="tokenization path")
    parser.add_argument("--example_cache_dir", default="EKMRC/data/ReCoRD_graph/tmp", 
                        type=str, help="example cache dir path")
    
    # log data path（use default dir runs）
    # parser.add_argument("--save_log_dir", default="EKMRC/log_dir", type=str, help="saved log dir")
    parser.add_argument("--log_comment", default="skgnet", type=str, help="log file comment")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=False,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--freeze_bert", action='store_true', 
                        help="If true, all of the BERT parameters are freezed")

    parser.add_argument('--logging_steps', type=int, default=4275,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=4275,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', default=False,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    # if args.local_rank == -1 or args.no_cuda:
    #     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #     args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl')
    #     args.n_gpu = 1
    # args.device = device

    # set n_gpu num
    # args.n_gpu = 1

    # Setup logging
    # logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt = '%m/%d/%Y %H:%M:%S',
    #                     level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #                 args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    # set_seed(args)

    # Load pretrained model and tokenizer
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    
    # process train and dev data
    if args.do_train:
        train_dataset, train_entity_emb_mat, train_relation_emb_mat = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)

    # if args.do_eval:
    #     _, entity_emb_mat, relation_emb_mat = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    
    logger.info("Process all data done!")

if __name__ == "__main__":
    main()