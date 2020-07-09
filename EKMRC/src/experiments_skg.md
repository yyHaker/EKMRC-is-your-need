# experiments info

## ReCoRD experiments

## train skg-net models (conceptnet)
CUDA_VISIBLE_DEVICES=0 python EKMRC/run_record_skg.py \
  --model_type skgnet \
  --model_name_or_path bert-large-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_conceptnet \
  --train_file EKMRC/data/ReCoRD_graph/train.json \
  --predict_file EKMRC/data/ReCoRD_graph/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment skgnet_large_conceptnet \
  --output_dir EKMRC/skg_results/skg_large_conceptnet/ \
  --overwrite_output_dir


## load data skg examples
CUDA_VISIBLE_DEVICES=5 python EKMRC/load_cache_record_skg_examples.py \
  --model_type skgnet \
  --model_name_or_path bert-large-cased\
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_conceptnet \
  --train_file EKMRC/data/ReCoRD/train.json \
  --predict_file EKMRC/data/ReCoRD/dev.json