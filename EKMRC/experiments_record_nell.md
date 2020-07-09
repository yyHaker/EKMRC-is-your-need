# FSNET models on ReCoRD, add NELL，for one-stage finetuning.

## FSNET(bert-large-uncased)
CUDA_VISIBLE_DEVICES=5 python EKMRC/run_record_concept.py \
  --model_type fsnet \
  --model_name_or_path bert-large-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --tokenization_path EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased \
  --embedding_path EKMRC/kgs/NELL/nell_emb/nell_concept2vec.txt \
  --retrieved_nell_concept_path EKMRC/kgs/NELL/retrieve_nell_res/bert-large-uncased/ \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment fsnet_bert_large_uncased_record_nell \
  --output_dir EKMRC/results/nell_record/debug_fsnet_bert_large_uncased/ \
  --overwrite_output_dir


## FSNET(bert-large-uncased-whole-word-masking)
CUDA_VISIBLE_DEVICES=4 python EKMRC/run_record_concept.py \
  --model_type fsnet \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --tokenization_path EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm \
  --embedding_path EKMRC/kgs/NELL/nell_emb/nell_concept2vec.txt \
  --retrieved_nell_concept_path EKMRC/kgs/NELL/retrieve_nell_res/bert-large-uncased-wwm/ \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment fsnet_bert_large_uncased_wwm_record_nell \
  --output_dir EKMRC/results/nell_record/debug_fsnet_bert_large_uncased_wwm/ \
  --overwrite_output_dir


# FSNET models on ReCoRD, add NELL，for two-stage finetuning.

## FSNET(bert-large-uncased-whole-word-masking)-stage1
CUDA_VISIBLE_DEVICES=1 python EKMRC/run_record_concept.py \
  --model_type fsnet \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_bert \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --tokenization_path EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm \
  --embedding_path EKMRC/kgs/NELL/nell_emb/nell_concept2vec.txt \
  --retrieved_nell_concept_path EKMRC/kgs/NELL/retrieve_nell_res/bert-large-uncased-wwm/ \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-6 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment fsnet_bert_large_uncased_wwm_record_nell_stage1_lr3e-6 \
  --output_dir EKMRC/results/nell_record/two_stages/debug_fsnet_bert_large_uncased_wwm_stage1_lr3e-6/ \
  --overwrite_output_dir

## FSNET(bert-large-uncased-whole-word-masking)-stage2
CUDA_VISIBLE_DEVICES=4 python EKMRC/run_record_concept.py \
  --model_type fsnet \
  --model_name_or_path EKMRC/results/nell_record/two_stages/debug_fsnet_bert_large_uncased_wwm_stage1_lr3e-4 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --tokenization_path EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm \
  --embedding_path EKMRC/kgs/NELL/nell_emb/nell_concept2vec.txt \
  --retrieved_nell_concept_path EKMRC/kgs/NELL/retrieve_nell_res/bert-large-uncased-wwm/ \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment fsnet_bert_large_uncased_wwm_record_nell_stage1_lr3e-4_stage2_lr3e-5 \
  --output_dir EKMRC/results/nell_record/two_stages/debug_fsnet_bert_large_uncased_wwm_stage1_lr3e-4_stage2_lr3e-5/ \
  --overwrite_output_dir

