# for FS-NET(bert-large-uncased-wwm)

## do tokenization on ReCoRD
python EKMRC/tokenization_record/do_tokenization.py \
    --train_file EKMRC/data/ReCoRD/train.json \
    --dev_file EKMRC/data/ReCoRD/dev.json \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --output_dir EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm \
    --do_lower_case


## retrieve NELL concept
python EKMRC/preprocess/retrieve_concepts/retrieve_nell/retrieve.py \
    --train_token EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm/train.tokenization.uncased.data \
    --eval_token EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm/dev.tokenization.uncased.data \
    --output_dir EKMRC/kgs/NELL/retrieve_nell_res/bert-large-uncased-wwm

## retrieve WordNet concept
python EKMRC/preprocess/retrieve_concepts/retrieve_wordnet/retrieve.py \
    --train_token EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm/train.tokenization.uncased.data \
    --eval_token EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased-wwm/dev.tokenization.uncased.data \
    --output_dir EKMRC/kgs/wordnet/retrieve_wordnet_res/bert-large-uncased-wwm


# for FS-NET(bert-large-uncased)

## do tokenization on ReCoRD
python EKMRC/preprocess/tokenization_record/do_tokenization.py \
    --train_file EKMRC/data/ReCoRD/train.json \
    --dev_file EKMRC/data/ReCoRD/dev.json \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --output_dir EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased \
    --do_lower_case


## retrieve NELL concept
python EKMRC/preprocess/retrieve_concepts/retrieve_nell/retrieve.py \
    --train_token EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased/train.tokenization.uncased.data \
    --eval_token EKMRC/data/ReCoRD/tokenization_self/bert-large-uncased/dev.tokenization.uncased.data \
    --output_dir EKMRC/kgs/NELL/retrieve_nell_res/bert-large-uncased

