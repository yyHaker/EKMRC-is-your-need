# pretrained models on ReCoRD

## bert-base-uncased
CUDA_VISIBLE_DEVICES=2 python EKMRC/run_record.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file EKMRC/data/ReCoRD_pure/train.json \
  --predict_file EKMRC/data/ReCoRD_pure/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment bert_base_uncased_record_pure \
  --output_dir EKMRC/results/pure_record/debug_bert_base_uncased \
  --overwrite_output_dir

## bert-large-uncased
CUDA_VISIBLE_DEVICES=0 python EKMRC/run_record.py \
  --model_type bert \
  --model_name_or_path bert-large-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file EKMRC/data/ReCoRD_pure/train.json \
  --predict_file EKMRC/data/ReCoRD_pure/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment bert_large_uncased_record_pure \
  --output_dir EKMRC/results/pure_record/debug_bert_large_uncased \
  --overwrite_output_dir

## bert-large-uncased-wwm
CUDA_VISIBLE_DEVICES=1 python EKMRC/run_record.py \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file EKMRC/data/ReCoRD_pure/train.json \
  --predict_file EKMRC/data/ReCoRD_pure/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment bert_large_uncased_wwm_record_pure \
  --output_dir EKMRC/results/pure_record/debug_bert_large_uncased_wwm \
  --overwrite_output_dir