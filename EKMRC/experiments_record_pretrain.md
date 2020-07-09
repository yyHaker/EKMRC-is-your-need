# experiments info

## ReCoRD experiments

### train KT-net(base) models (nell)
CUDA_VISIBLE_DEVICES=1 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_bert \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment ktnet_base_nell_stage1 \
  --output_dir EKMRC/results/debug_ktnet_base_cpnet_nell_stage1/ \
  --overwrite_output_dir

### train KT-net(large) models (nell)
CUDA_VISIBLE_DEVICES=0 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path bert-large-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --freeze_bert \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --evaluate_during_training \
  --learning_rate 3e-4 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment ktnet_large_nell_stage1 \
  --output_dir EKMRC/results/two_stages/debug_ktnet_large_cpnet_nell_stage1/ \
  --overwrite_output_dir
  
### train KT-net(base) models (wordnet) 
CUDA_VISIBLE_DEVICES=1 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_wordnet \
  --freeze_bert \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment ktnet_base_wordnet_stage1 \
  --output_dir EKMRC/results/two_stages/debug_ktnet_base_cpnet_wordnet_stage1/ \
  --overwrite_output_dir

### train KT-net(large) models (wordnet) (7GB)
CUDA_VISIBLE_DEVICES=0 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path bert-large-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_wordnet \
  --freeze_bert \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --evaluate_during_training \
  --learning_rate 3e-4 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment ktnet_large_wordnet_stage1 \
  --output_dir EKMRC/results/two_stages/debug_ktnet_large_cpnet_wordnet_stage1/ \
  --overwrite_output_dir

### train KT-net(base) models (wordnet + nell)
CUDA_VISIBLE_DEVICES=2 python EKMRC/run_record_concept_both.py \
  --model_type ktnet \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_wordnet \
  --use_nell \
  --freeze_bert \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment ktnet_base_wordnet_nell_stage1 \
  --output_dir EKMRC/results/debug_ktnet_base_wordnet_nell_stage1/ \
  --overwrite_output_dir

### train KT-net(large) models (wordnet + nell)
CUDA_VISIBLE_DEVICES=0 python EKMRC/run_record_concept_both.py \
  --model_type ktnet \
  --model_name_or_path bert-large-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_wordnet \
  --use_nell \
  --freeze_bert \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --evaluate_during_training \
  --learning_rate 3e-4 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment ktnet_large_wordnet_nell_stage1 \
  --output_dir EKMRC/results/two_stages/debug_ktnet_large_wordnet_nell_stage1/ \
  --overwrite_output_dir

### just eval trained ktnet model
CUDA_VISIBLE_DEVICES=1 python EKMRC/run_record_concept.py \
  --model_type ktnet \
  --model_name_or_path bert-large-cased \
  --do_eval \
  --do_lower_case \
  --use_nell \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment ktnet_large_nell \
  --output_dir EKMRC/results/debug_ktnet_large_cpnet_nell/ \
  --overwrite_output_dir

### just fine-tuning BERT(base) on ReCoRD
CUDA_VISIBLE_DEVICES=2 python EKMRC/run_record.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment bert_base \
  --output_dir EKMRC/results/debug_bert_base/ \
  --overwrite_output_dir

### just fine-tuning BERT(large) on ReCoRD
CUDA_VISIBLE_DEVICES=0 python EKMRC/run_record.py \
  --model_type bert \
  --model_name_or_path bert-large-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment bert_large \
  --output_dir EKMRC/results/debug_bert_large/ \
  --overwrite_output_dir

## SQuAD1.1
CUDA_VISIBLE_DEVICES=2,3 python EKMRC/run_squad.py \
  --model_type bert \
  --model_name_or_path output_cpnet_lm \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file EKMRC/data/squad/train-v1.1.json \
  --predict_file EKMRC/data/squad/dev-v1.1.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-6 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir EKMRC/results/debug_squad/ \
  --overwrite_output_dir