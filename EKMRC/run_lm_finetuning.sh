export TRAIN_FILE=EKMRC/data/conceptnet/conceptnet-assertions-5.6.0.csv.en.train.entity.context.txt
export TEST_FILE=EKMRC/data/conceptnet/conceptnet-assertions-5.6.0.csv.en.dev.entity.context.txt

CUDA_VISIBLE_DEVICES=1,3 python EKMRC/run_lm_finetuning.py \
    --output_dir=output_cpnet_entity_context_lm \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --overwrite_output_dir