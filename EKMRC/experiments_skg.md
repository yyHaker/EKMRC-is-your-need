# experiments info

## ReCoRD experiments

## train skg-net models (conceptnet)
CUDA_VISIBLE_DEVICES=4 python EKMRC/run_record_skg.py \
  --model_type skgnet \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --use_conceptnet \
  --train_file EKMRC/data/ReCoRD_graph/train.json \
  --predict_file EKMRC/data/ReCoRD_graph/dev.json \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --log_comment skgnet_base_conceptnet \
  --output_dir EKMRC/skg_results/debug_skg_base_conceptnet/ \
  --overwrite_output_dir
