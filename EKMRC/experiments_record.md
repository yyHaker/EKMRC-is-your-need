# one stage experiments here


## train FS-net(large) models (nell)
CUDA_VISIBLE_DEVICES=5 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path bert-large-cased\
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD_new/train.json \
  --predict_file EKMRC/data/ReCoRD_new/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --scheduler cosine \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment ktnet_large_nell_cosine \
  --output_dir EKMRC/results/one_stage/ktnet_large_nell_cosine/ \
  --overwrite_output_dir

CUDA_VISIBLE_DEVICES=2 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path bert-large-cased\
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD_new/train.json \
  --predict_file EKMRC/data/ReCoRD_new/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --scheduler linear \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment ktnet_large_nell_linear \
  --output_dir EKMRC/results/one_stage/ktnet_large_nell_linear/ \
  --overwrite_output_dir

## pretrain FS-NET(large) models (nell)
CUDA_VISIBLE_DEVICES=0 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path bert-large-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_bert \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD_new/train.json \
  --predict_file EKMRC/data/ReCoRD_new/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --evaluate_during_training \
  --scheduler linear \
  --learning_rate 3e-4 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment FSNET_large_nell_fixemb_stage1 \
  --output_dir EKMRC/results/two_stages/FSNET_large_nell_fixemb_stage1/ \
  --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path EKMRC/results/two_stages/FSNET_large_nell_fixemb_stage1/ \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_concept \
  --train_file EKMRC/data/ReCoRD_new/train.json \
  --predict_file EKMRC/data/ReCoRD_new/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --scheduler linear \
  --learning_rate 3e-4 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment FSNET_large_nell_fixemb_stage2 \
  --output_dir EKMRC/results/two_stages/FSNET_large_nell_fixemb_stage2/ \
  --overwrite_output_dir


CUDA_VISIBLE_DEVICES=2 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path bert-large-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_bert \
  --train_file EKMRC/data/ReCoRD_new/train.json \
  --predict_file EKMRC/data/ReCoRD_new/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --evaluate_during_training \
  --scheduler linear \
  --learning_rate 3e-4 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment FSNET_large_nell_updateEmb_stage1 \
  --output_dir EKMRC/results/two_stages/FSNET_large_nell_updateEmb_stage1/ \
  --overwrite_output_dir

  # bert-large(wwm) model
  CUDA_VISIBLE_DEVICES=4 python EKMRC/run_record.py \
  --model_type bert \
  --model_name_or_path /users4/yle/pretrained_models/bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --gradient_accumulation_steps 6 \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment bert_large_wwm_record \
  --output_dir EKMRC/results/debug_bert_large_wwm_record/ \
  --overwrite_output_dir

## FS-net bert-large(wwm) model
CUDA_VISIBLE_DEVICES=5 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path /users4/yle/pretrained_models/bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment fs_net_bert_large_wwm_record \
  --output_dir EKMRC/results/debug_fs_net_bert_large_wwm_record/ \
  --overwrite_output_dir



