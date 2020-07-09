# add NER multi-task

## FS-net(bert-large-uncased-whole-word-masking) + NELL model
CUDA_VISIBLE_DEVICES=0 python EKMRC/run_record_concept_ner.py \
  --model_type fsnet_ner \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD_ner/train.json \
  --predict_file EKMRC/data/ReCoRD_ner/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment FSNET_ner_bert_large_wwm_record_lr3e-5 \
  --output_dir EKMRC/results/record_ner_nell/debug_FSNET_ner_bert_large_wwm_record_lr3e-5/ \
  --overwrite_output_dir


## FSNER-net(bert-large-uncased-whole-word-masking) + NELL model + two-stages

# stage-1
CUDA_VISIBLE_DEVICES=4 python EKMRC/run_record_concept_ner.py \
  --model_type fsnet_ner \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_bert \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD_ner/train.json \
  --predict_file EKMRC/data/ReCoRD_ner/dev.json \
  --tokenization_path EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm \
  --embedding_path EKMRC/kgs/NELL/nell_emb/nell_concept2vec.txt \
  --retrieved_nell_concept_path EKMRC/kgs/NELL/retrieve_nell_res/bert-large-uncased-wwm/ \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-4 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment fsnet_ner_bert_large_uncased_wwm_record_nell_stage1_lr3e-4 \
  --output_dir EKMRC/results/record_ner_nell/two_stages/debug_fsnet_ner_bert_large_uncased_wwm_stage1_lr3e-4/ \
  --overwrite_output_dir

# stage-2
CUDA_VISIBLE_DEVICES=2 python EKMRC/run_record_concept_ner.py \
  --model_type fsnet_ner \
  --model_name_or_path EKMRC/results/record_ner_nell/two_stages/debug_fsnet_ner_bert_large_uncased_wwm_stage1_lr3e-4/ \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD_ner/train.json \
  --predict_file EKMRC/data/ReCoRD_ner/dev.json \
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
  --log_comment fsnet_ner_bert_large_uncased_wwm_record_nell_stage2_lr3e-5 \
  --output_dir EKMRC/results/record_ner_nell/two_stages/debug_fsnet_ner_bert_large_uncased_wwm_stage2_lr3e-5/ \
  --overwrite_output_dir
